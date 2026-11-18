"""Experimentos rápidos sobre los datos: ¿alguna fuente mete ruido? ¿qué ventana conviene?

Entrena redes conjuntas (por defecto la RNN bidireccional y la conv + BiLSTM) cambiando solo
los datos de entrenamiento, y las evalúa SIEMPRE sobre el mismo conjunto fijo:

- Cada documento de cada fuente se corta en bloques de oraciones seguidas (~300 palabras) y
  ~20% de los bloques se reservan para evaluación. Los bloques se separan ANTES de cortar en
  ventanas, así ninguna ventana de train comparte texto con eval, sea cual sea la ventana.
- Eval se corta siempre con la ventana de referencia (5-60 uniforme), así todas las
  configuraciones se miden sobre exactamente las mismas instancias.
- Train usa un presupuesto de PALABRAS (no de instancias), así cambiar la ventana o sacar una
  fuente no cambia cuánto texto ve la red. Con --fixed-total (default) el presupuesto total
  se mantiene al sacar una fuente: las demás aportan más.
- La época se elige sobre un 10% de validación sacado de train, nunca sobre eval.

Modos:

    # leave-one-source-out: todas las fuentes y luego cada una sacada
    uv run python -m src.data_ablation sources --epochs 5

    # ventanas: MIN-MAX:distribución, un N fijo, o 0 = una oración por instancia
    uv run python -m src.data_ablation windows --epochs 5 \\
        --windows 5-60:uniform 5-60:normal 5-100:uniform 5-150:uniform 20-100:uniform 30-150:uniform

Cómo leer `sources`: sacar la fuente X casi siempre baja el F1 sobre X (no la vio). La
pregunta es qué pasa con el F1 sobre las DEMÁS fuentes (columna `others`): si sube al sacar
X, X está metiendo ruido. Diferencias menores a la variación entre semillas (--seeds) no
significan nada.
"""

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

from src.constants import MODELS_DIR, RANDOM_SEED
from src.data import JointDataset, build_loader, load_documents
from src.data.samples import Sample, WindowSpec, build_samples
from src.data.sources import Document
from src.features import EmbeddingsExtractor
from src.metrics import JointMetric
from src.metrics.base import scores_from_tasks
from src.problem.joint import create_conv_bilstm_model, create_joint_model
from src.ablation_plots import save_summary_plots
from src.metrics.base import WordLevelTask
from src.training.neural import evaluate_neural_model, train_neural_model
from src.training.plots import save_confusion_matrices, save_training_plot
from src.training.result import TrainResult, format_score, mean_score

logger = logging.getLogger("src.data_ablation")

MODELS = {
    "birnn": lambda: create_joint_model(bidirectional=True),
    "bilstm": lambda: create_joint_model(bidirectional=True, lstm=True),
    "conv_bilstm": create_conv_bilstm_model,
}
CHUNK_WORDS = 300
EVAL_RATIO = 0.2
VALIDATION_RATIO = 0.1
REFERENCE_WINDOW = WindowSpec()  # 5-60 uniforme, la del pipeline


@dataclass
class SourceSplit:
    train_chunks: list[Document]  # en orden aleatorio (fijo por semilla)
    eval_samples: list[Sample]


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Agrupa oraciones seguidas de cada documento en bloques de ~CHUNK_WORDS palabras"""
    chunks = []
    for document in documents:
        current, n_words = [], 0
        for sentence in document:
            current.append(sentence)
            n_words += len(sentence.split())
            if n_words >= CHUNK_WORDS:
                chunks.append(current)
                current, n_words = [], 0
        if current:
            chunks.append(current)
    return chunks


def take_words(samples: list[Sample], budget: int) -> list[Sample]:
    """Las primeras instancias hasta juntar `budget` palabras"""
    taken, n_words = [], 0
    for sample in samples:
        if n_words >= budget:
            break
        taken.append(sample)
        n_words += len(sample)
    return taken


def split_sources(documents: dict[str, list[Document]], eval_words: int) -> dict[str, SourceSplit]:
    """Separa los bloques de cada fuente en train y eval, siempre igual (semilla fija)"""
    splits = {}
    for i, (name, source_documents) in enumerate(documents.items()):
        rng = np.random.default_rng(RANDOM_SEED + i)
        chunks = chunk_documents(source_documents)
        order = rng.permutation(len(chunks))
        n_eval = max(1, int(EVAL_RATIO * len(chunks)))
        eval_chunks = [chunks[j] for j in order[:n_eval]]
        train_chunks = [chunks[j] for j in order[n_eval:]]
        eval_samples = take_words(build_samples(eval_chunks, REFERENCE_WINDOW), eval_words)
        splits[name] = SourceSplit(train_chunks, eval_samples)
        logger.info(
            "Fuente %s: %d bloques train, eval=%d instancias (%d palabras)",
            name,
            len(train_chunks),
            len(eval_samples),
            sum(map(len, eval_samples)),
        )
    return splits


def train_samples(
    splits: dict[str, SourceSplit], sources: list[str], window: WindowSpec | None, words_per_source: int
) -> list[Sample]:
    samples = []
    for name in sources:
        source_samples = take_words(build_samples(splits[name].train_chunks, window), words_per_source)
        if sum(map(len, source_samples)) < words_per_source * 0.95:
            logger.warning("Fuente %s no alcanza %d palabras de train", name, words_per_source)
        samples.extend(source_samples)
    return samples


def build_model(model_name: str, seed: int) -> torch.nn.Module:
    """La red del pipeline. Con otra semilla se reinicializan los pesos (y cambia el shuffle)"""
    model = MODELS[model_name]()
    torch.manual_seed(seed)
    if seed != RANDOM_SEED:
        for module in model.modules():
            if module is not model and hasattr(module, "reset_parameters"):
                module.reset_parameters()
    return model


def run_one(
    samples: list[Sample],
    eval_sets: dict[str, JointDataset],
    model_name: str,
    seed: int,
    epochs: int,
    extractor: EmbeddingsExtractor,
) -> tuple[TrainResult, dict[str, dict[str, WordLevelTask]]]:
    """Entrena una red con `samples`. Devuelve el resultado (red, historial por época sobre
    train y validación) y las predicciones a nivel palabra sobre cada conjunto de eval"""
    dataset = JointDataset(samples)
    train_set, validation_set = random_split(
        dataset, [1 - VALIDATION_RATIO, VALIDATION_RATIO], generator=torch.Generator().manual_seed(seed)
    )
    model = build_model(model_name, seed)
    result = train_neural_model(
        build_loader(train_set, shuffle=True),
        build_loader(validation_set, shuffle=False),
        extractor,
        JointMetric(),
        model,
        n_epochs=epochs,
    )
    tasks = {
        name: evaluate_neural_model(result.model, build_loader(eval_set, False), extractor, JointMetric())
        for name, eval_set in eval_sets.items()
    }
    return result, tasks


def safe_name(label: str) -> str:
    """'5-60:uniform' -> '5-60_uniform', '-ted' -> 'sin_ted' (para nombres de archivo)"""
    return label.replace(":", "_").replace("-", "sin_", 1) if label.startswith("-") else label.replace(":", "_")


def save_run(run_dir: Path, model_name: str, label: str, seed: int, result: TrainResult) -> None:
    """Pesos (.pkl) y gráfico de la evolución por época (train y validación) de una red"""
    stem = run_dir / "runs" / model_name / safe_name(label) / f"seed{seed}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), stem.with_suffix(".pkl"))
    save_training_plot(
        result.history, stem.with_suffix(".png"), f"{model_name} · {label} · seed {seed} (train y validación)",
        result.best_epoch,
    )


def history_rows(model_name: str, label: str, seed: int, result: TrainResult) -> list[list]:
    """Una fila por época y tarea: loss y F1 de train y de validación"""
    return [
        [model_name, label, seed, m.epoch, m.loss, m.test_loss, task, m.score[task], m.test_score[task]]
        for m in result.history
        for task in m.score
    ]


def parse_window(text: str) -> WindowSpec | None:
    """'5-60:normal', '5-60' (uniforme), '10' (fijo) o '0' (por oración)"""
    size, _, distribution = text.partition(":")
    if size == "0":
        return None
    low, _, high = size.partition("-")
    return WindowSpec(int(low), int(high or low), distribution or "uniform")


def window_label(window: WindowSpec | None) -> str:
    if window is None:
        return "sentence"
    if window.min_size == window.max_size:
        return f"fixed{window.min_size}"
    return f"{window.min_size}-{window.max_size}:{window.distribution}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["sources", "windows", "plot"])
    parser.add_argument("run_dir", nargs="?", help="en modo plot: la carpeta de la corrida (models/ablation/...)")
    parser.add_argument("--models", type=lambda s: s.split(","), default=["birnn", "conv_bilstm"], help=f"de {list(MODELS)}")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[RANDOM_SEED], help="varias para medir el ruido")
    parser.add_argument("--words-per-source", type=int, default=20000, help="palabras de train por fuente")
    parser.add_argument("--eval-words", type=int, default=5000, help="palabras de eval por fuente")
    parser.add_argument("--no-fixed-total", action="store_true", help="al sacar una fuente, no compensar con las demás")
    parser.add_argument("--sources", type=lambda s: s.split(","), default=None, help="fuentes a sacar (default: todas)")
    parser.add_argument("--windows", nargs="+", default=["5-60:uniform", "5-60:normal", "5-100:uniform", "5-150:uniform", "20-100:uniform", "30-150:uniform"])
    parser.add_argument("--window", default="5-60:uniform", help="ventana de train en modo sources")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("src").setLevel(logging.INFO)
    logging.getLogger("src.training.neural").setLevel(logging.WARNING)  # sin el log de cada batch

    if args.mode == "plot":
        if not args.run_dir:
            parser.error("plot necesita la carpeta de la corrida")
        for path in save_summary_plots(Path(args.run_dir)):
            print(f"Guardado {path}")
        return

    unknown = set(args.models) - set(MODELS)
    if unknown:
        parser.error(f"modelos desconocidos {sorted(unknown)}, opciones: {list(MODELS)}")

    documents = load_documents()
    all_sources = list(documents)
    splits = split_sources(documents, args.eval_words)
    eval_sets = {name: JointDataset(split.eval_samples) for name, split in splits.items()}
    eval_sets["all"] = JointDataset([s for split in splits.values() for s in split.eval_samples])
    extractor = EmbeddingsExtractor()

    # (etiqueta, fuentes de train, ventana, palabras por fuente)
    configs = []
    if args.mode == "sources":
        window = parse_window(args.window)
        configs.append(("all", all_sources, window, args.words_per_source))
        for dropped in args.sources or all_sources:
            kept = [s for s in all_sources if s != dropped]
            budget = args.words_per_source
            if not args.no_fixed_total:
                budget = round(budget * len(all_sources) / len(kept))
            configs.append((f"-{dropped}", kept, window, budget))
    else:
        for text in args.windows:
            window = parse_window(text)
            configs.append((window_label(window), all_sources, window, args.words_per_source))

    run_dir = Path(MODELS_DIR) / "ablation" / f"{args.mode}__{datetime.now():%Y-%m-%d__%H-%M-%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Guardando todo en %s", run_dir)

    # Con qué se corrió: argumentos, y fuentes/ventana/tamaño de train de cada config
    config_info = {"args": vars(args), "reference_eval_window": asdict(REFERENCE_WINDOW), "chunk_words": CHUNK_WORDS,
                   "eval_ratio": EVAL_RATIO, "validation_ratio": VALIDATION_RATIO,
                   "eval": {name: {"instances": len(s.eval_samples), "words": sum(map(len, s.eval_samples))}
                            for name, s in splits.items()},
                   "configs": {}}
    train_sets = {}
    for label, sources, window, budget in configs:
        samples = train_samples(splits, sources, window, budget)
        train_sets[label] = samples
        config_info["configs"][label] = {"sources": sources, "window": asdict(window) if window else None,
                                         "words_per_source": budget, "instances": len(samples),
                                         "words": sum(map(len, samples))}
    (run_dir / "config.json").write_text(json.dumps(config_info, indent=2, ensure_ascii=False))

    rows = []
    with (run_dir / "results.csv").open("w", newline="") as f, (run_dir / "history.csv").open("w", newline="") as h:
        writer, history_writer = csv.writer(f), csv.writer(h)
        writer.writerow(["mode", "config", "model", "seed", "best_epoch", "eval_set", "task", "f1"])
        history_writer.writerow(
            ["model", "config", "seed", "epoch", "train_loss", "val_loss", "task", "train_f1", "val_f1"]
        )
        for model_name in args.models:
            confusion = {}  # config -> predicciones sobre todo eval de la primera semilla
            for label, *_ in configs:
                samples = train_sets[label]
                for seed in args.seeds:
                    start = time.time()
                    result, tasks = run_one(samples, eval_sets, model_name, seed, args.epochs, extractor)
                    scores = {name: scores_from_tasks(t) for name, t in tasks.items()}
                    logger.info(
                        "%s %s seed=%d (%d inst., %d palabras, %.0fs, época %d/%d) eval all: %s",
                        model_name, label, seed, len(samples), sum(map(len, samples)),
                        time.time() - start, result.best_epoch, args.epochs, format_score(scores["all"]),
                    )
                    for eval_name, task_scores in scores.items():
                        for task, f1 in task_scores.items():
                            row = [args.mode, label, model_name, seed, result.best_epoch, eval_name, task, f1]
                            writer.writerow(row)
                            rows.append(row)
                    history_writer.writerows(history_rows(model_name, label, seed, result))
                    f.flush()
                    h.flush()
                    save_run(run_dir, model_name, label, seed, result)
                    if seed == args.seeds[0]:
                        confusion[label] = TrainResult(None, tasks["all"])
            save_confusion_matrices(
                confusion, run_dir / f"confusion_{model_name}.png",
                f"{model_name} · matrices de confusión sobre todo eval (seed {args.seeds[0]})",
            )

    print_summary(rows, configs, all_sources, args.mode)
    for path in save_summary_plots(run_dir):
        logger.info("Gráfico %s", path)
    print(f"\nTodo guardado en {run_dir}")


def print_summary(rows: list[list], configs: list[tuple], all_sources: list[str], mode: str) -> None:
    """F1 promedio de las 3 tareas (media entre semillas) y su diferencia con la primera config"""
    mean = {}  # (model, config, eval_set) -> F1 promedio de tareas y semillas
    for model, config, eval_set in {(r[2], r[1], r[5]) for r in rows}:
        per_seed = {}
        for r in rows:
            if (r[2], r[1], r[5]) == (model, config, eval_set):
                per_seed.setdefault(r[3], {})[r[6]] = r[7]
        values = [mean_score(score) for score in per_seed.values()]
        mean[model, config, eval_set] = (float(np.mean(values)), float(np.std(values)))

    columns = ["all"] + (["others"] if mode == "sources" else []) + all_sources
    for model in dict.fromkeys(r[2] for r in rows):
        print(f"\n=== {model}: F1 macro promedio (cap, punt_inicial, punt_final) sobre eval; Δ vs {configs[0][0]} ===")
        print(f"{'config':<22}" + "".join(f"{c:>16}" for c in columns))
        baseline = configs[0][0]
        for label, *_ in configs:
            cells = []
            for column in columns:
                if column == "others":  # promedio de las fuentes que SÍ estaban en train
                    eval_names = [s for s in all_sources if f"-{s}" != label]
                    value = np.mean([mean[model, label, s][0] for s in eval_names])
                    base = np.mean([mean[model, baseline, s][0] for s in eval_names])
                else:
                    value, base = mean[model, label, column][0], mean[model, baseline, column][0]
                cells.append(f"{value:.3f}" if label == baseline else f"{value:.3f}({value - base:+.3f})")
            print(f"{label:<22}" + "".join(f"{c:>16}" for c in cells))
        stds = [mean[model, label, "all"][1] for label, *_ in configs]
        if max(stds) > 0:
            print(f"desvío entre semillas en 'all': hasta {max(stds):.3f} (diferencias menores son ruido)")


if __name__ == "__main__":
    main()
