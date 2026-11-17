"""Barrido de regularización sobre la RNN bidireccional de cada problema.

Entrena `bidirectional_rnn` (capitalización y puntuación) una vez por cada valor de un
hiperparámetro, con todo lo demás fijo (datos, splits, semilla, épocas):

- `--param dropout`: probabilidad de dropout sobre la salida de la RNN.
- `--param weight_decay`: weight decay de AdamW (0 = Adam sin regularización).

Guarda en `models/regularization/<día>__<hora>__<param>/results.json` el historial por época
(loss y F1 de train y test) y el F1 sobre test y control de cada valor. El archivo se
reescribe después de cada entrenamiento, así que si Colab se corta queda lo que ya terminó.
Los notebooks `regularization_*.ipynb` de esta carpeta lo corren y grafican. Se corre desde la raíz del repo:

    uv run python -m notebooks.hyperparameters.regularization.regularization --param dropout --values 0 0.1 0.3 0.5
    uv run python -m notebooks.hyperparameters.regularization.regularization --param weight_decay --values 0 1e-3 1e-2 1e-1 0.5 1
"""

import json
import logging
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Dataset

from src.constants import CAPITALIZATION_DATASET_SIZE, MODELS_DIR, PUNCTUATION_DATASET_SIZE, RUN_SEPARATOR
from src.data import build_loader, preprocess_dataset, split_dataset_dev_control, split_train_test
from src.data.samples import add_window_args, window_from_args
from src.features import EmbeddingsExtractor
from src.problem import CapitalizationProblem, PunctuationProblem, Problem
from src.training import neural
from src.training.neural import evaluate_problem_network, train_neural_model
from src.training.result import TrainResult, format_score
from src.metrics.base import scores_from_tasks

logger = logging.getLogger("src.experiments.regularization")  # bajo "src" para que salga en INFO

PARAMS = ("dropout", "weight_decay")


def parse_args() -> Namespace:
    """Lee las opciones del barrido.

    Returns
    -------
    Namespace
        Hiperparámetro a variar, sus valores, los valores fijos, épocas, tamaños y ventana.
    """
    parser = ArgumentParser(description="Barrido de dropout o weight decay sobre la RNN bidireccional")
    parser.add_argument("--param", choices=PARAMS, required=True, help="hiperparámetro a variar")
    parser.add_argument("--values", type=float, nargs="+", required=True, help="valores a probar")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout fijo al variar weight_decay (default: 0)")
    parser.add_argument(
        "--weight-decay", type=float, default=0.0, help="weight decay fijo al variar dropout (default: 0)"
    )
    parser.add_argument("--epochs", type=int, default=neural.N_EPOCHS, help=f"épocas (default: {neural.N_EPOCHS})")
    parser.add_argument("--capitalization-size", type=int, default=CAPITALIZATION_DATASET_SIZE)
    parser.add_argument("--punctuation-size", type=int, default=PUNCTUATION_DATASET_SIZE)
    add_window_args(parser)
    return parser.parse_args()


def _loaders(dataset: Dataset) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Crea los loaders de train, test y control con los mismos splits y semillas que el pipeline.

    Parameters
    ----------
    dataset : Dataset
        Dataset completo del problema.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        Loaders de train (mezclado), test y control.
    """
    development_set, control_set = split_dataset_dev_control(dataset)
    train_set, test_set = split_train_test(development_set)
    logger.info("Split: train=%d, test=%d, control=%d", len(train_set), len(test_set), len(control_set))
    return (
        build_loader(train_set, shuffle=True),
        build_loader(test_set, shuffle=False),
        build_loader(control_set, shuffle=False),
    )


def train_one(
    problem: Problem,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    extractor: EmbeddingsExtractor,
    args: Namespace,
    value: float,
) -> dict[str, Any]:
    """Entrena la red bidireccional con ``args.param = value`` y devuelve sus métricas.

    Parameters
    ----------
    problem : Problem
        Problema a resolver.
    loaders : tuple[DataLoader, DataLoader, DataLoader]
        Loaders de train, test y control.
    extractor : EmbeddingsExtractor
        Extractor de features por token.
    args : Namespace
        Argumentos del barrido (ver ``parse_args``).
    value : float
        Valor del hiperparámetro que se varía.

    Returns
    -------
    dict[str, Any]
        ``value``, ``dropout``, ``weight_decay`` y las métricas de ``_metrics``.
    """
    dropout = value if args.param == "dropout" else args.dropout
    weight_decay = value if args.param == "weight_decay" else args.weight_decay
    train_loader, test_loader, control_loader = loaders
    logger.info("%s: dropout=%g weight_decay=%g", type(problem).__name__, dropout, weight_decay)

    model = problem.bidirectional_rnn_model(dropout=dropout)
    result = train_neural_model(
        train_loader, test_loader, extractor, problem.metric, model, n_epochs=args.epochs, weight_decay=weight_decay
    )
    control = scores_from_tasks(evaluate_problem_network(result.model, control_loader, extractor, problem))
    logger.info("Control: %s", format_score(control))
    return {"value": value, "dropout": dropout, "weight_decay": weight_decay, **_metrics(result, control)}


def _metrics(result: TrainResult, control: dict[str, float]) -> dict[str, Any]:
    """Arma lo que se guarda de cada entrenamiento: historial, mejor época y F1 en test y control.

    Parameters
    ----------
    result : TrainResult
        Resultado del entrenamiento.
    control : dict[str, float]
        F1 macro de cada tarea sobre control.

    Returns
    -------
    dict[str, Any]
        ``{"best_epoch", "test_f1", "control_f1", "history"}``.
    """
    history = [
        {
            "epoch": m.epoch,
            "train_loss": m.loss,
            "test_loss": m.test_loss,
            "train_f1": m.score,
            "test_f1": m.test_score,
        }
        for m in result.history
    ]
    return {"best_epoch": result.best_epoch, "test_f1": result.score, "control_f1": control, "history": history}


def config(args: Namespace) -> dict[str, Any]:
    """Arma con qué se entrenó el barrido, para poder compararlo con otras corridas.

    Parameters
    ----------
    args : Namespace
        Argumentos del barrido.

    Returns
    -------
    dict[str, Any]
        Los argumentos, la ventana, el learning rate y el recorte del gradiente.
    """
    window = window_from_args(args)
    return {
        **{key: value for key, value in vars(args).items() if not key.startswith("window")},
        "window": asdict(window) if window else None,
        "learning_rate": neural.LEARNING_RATE,
        "grad_clip_norm": neural.GRAD_CLIP_NORM,
    }


def output_path(param: str) -> Path:
    """Crea la carpeta del barrido y devuelve dónde guardar sus resultados.

    Parameters
    ----------
    param : str
        ``"dropout"`` o ``"weight_decay"``.

    Returns
    -------
    Path
        ``models/regularization/<día>__<hora>__<param>/results.json``.
    """
    stamp = datetime.now().strftime(f"%Y-%m-%d{RUN_SEPARATOR}%H-%M-%S")
    run_dir = Path(MODELS_DIR) / "regularization" / RUN_SEPARATOR.join((stamp, param))
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "results.json"


def main() -> None:
    """Corre el barrido sobre los dos problemas, guardando los resultados después de cada entrenamiento."""
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("src").setLevel(logging.INFO)

    punctuation, capitalization = preprocess_dataset(
        args.punctuation_size, args.capitalization_size, window_from_args(args)
    )
    extractor, path = EmbeddingsExtractor(), output_path(args.param)
    report: dict[str, Any] = {"config": config(args), "problems": {}}
    for dataset, problem in [(capitalization, CapitalizationProblem()), (punctuation, PunctuationProblem())]:
        loaders = _loaders(dataset)
        runs = report["problems"].setdefault(type(problem).__name__, [])
        for value in args.values:
            runs.append(train_one(problem, loaders, extractor, args, value))
            path.write_text(json.dumps(report, indent=2))
            logger.info("Resultados guardados en %s", path)


if __name__ == "__main__":
    main()
