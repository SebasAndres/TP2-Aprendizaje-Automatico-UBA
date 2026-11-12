"""Prueba interactiva de los modelos guardados por el pipeline.

Pide un texto, lo normaliza igual que el dataset (minúsculas y sin puntuación), corre los
modelos y reconstruye el texto con la capitalización y la puntuación predichas.

Sin argumentos abre un menú para elegir los modelos guardados en models/ y probar textos:

    uv run python -m src.inference

Acepta redes (.pkl, solo los pesos) y modelos clásicos (.joblib, el RandomForest entero).
Cada modelo predice solo su problema; para reconstruir todo hay que elegir uno de cada
problema, o la red conjunta, que predice los dos. Los modelos también se
pueden pasar directamente, salteando el menú de modelos:

    uv run python -m src.inference -m models/CapitalizationProblem/bidirectional_rnn__2026-11-12_19-30__n5000.pkl \\
                                   -m models/PunctuationProblem/classic_embeddings__2026-11-12_19-30__n5000.joblib
    uv run python -m src.inference -m models/PunctuationProblem/joint_rnn__2026-11-12_19-30__n5000.pkl -v

Se usa como máximo un modelo por problema. Si falta el de un problema y el otro es la red
conjunta, la red conjunta predice los dos.
"""
from argparse import ArgumentParser
from dataclasses import dataclass
from collections import Counter
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import joblib
import questionary
import torch
import transformers
from torch import nn
from transformers import BertTokenizer

from src.constants import EMBEDDINGS_MODEL, MODELS_DIR, RUN_SEPARATOR
from src.data import strip_punctuation
from src.data.punctuation_dataset import FINAL_LABELS, INITIAL_LABELS
from src.features import (
    AverageEmbeddingsExtractor,
    ClassicFeatureExtractor,
    EmbeddingsExtractor,
    TextFeatureExtractor,
)
from src.joint_rnn import create_joint_rnn_model
from src.metrics.base import class_name
from src.problem import CapitalizationProblem, Problem, PunctuationProblem

PROBLEMS: dict[str, Problem] = {
    "CapitalizationProblem": CapitalizationProblem(),
    "PunctuationProblem": PunctuationProblem(),
}

# Mismos nombres y extractores que en build_experiments: el archivo se guarda con ese nombre
NETWORKS: dict[str, Callable[[Problem], nn.Module]] = {
    "standard_rnn": lambda problem: problem.standard_rnn_model(),
    "bidirectional_rnn": lambda problem: problem.bidirectional_rnn_model(),
    "joint_rnn": lambda _: create_joint_rnn_model(),
}
CLASSIC_EXTRACTORS: dict[str, type[TextFeatureExtractor]] = {
    "classic_features": ClassicFeatureExtractor,
    "classic_embeddings": AverageEmbeddingsExtractor,
}

Capitalizations = list[int]
"""Clase de capitalización de cada palabra: 0 minúscula, 1 inicial, 2 mixta, 3 mayúscula"""

Punctuations = list[tuple[int, int]]
"""(puntuación inicial, puntuación final) de cada palabra: índices de INITIAL_LABELS y FINAL_LABELS"""


def model_name(path: Path) -> str:
    """Nombre del modelo sin la fecha ni el tamaño del dataset de la corrida.
    Ej: 'standard_rnn__2026-11-12_19-30__n5000.pkl' -> 'standard_rnn'"""
    return path.stem.split(RUN_SEPARATOR)[0]


def run_info(path: Path) -> str:
    """Fecha y tamaño del dataset de la corrida, si el archivo los tiene. Ej: '2026-11-12_19-30 · n5000'"""
    return " · ".join(path.stem.split(RUN_SEPARATOR)[1:])


@cache
def _extractor(extractor_class: type[TextFeatureExtractor]) -> TextFeatureExtractor:
    """Una sola instancia por tipo de extractor: los de embeddings cargan BERT"""
    return extractor_class()


def _load(path: Path, problem: Problem, name: str) -> tuple[Any, type[TextFeatureExtractor]]:
    """Devuelve el modelo listo para predecir y el extractor con el que se entrenó"""
    if path.suffix == ".joblib" and name in CLASSIC_EXTRACTORS:
        return joblib.load(path), CLASSIC_EXTRACTORS[name]
    if path.suffix == ".pkl" and name in NETWORKS:
        model = NETWORKS[name](problem)
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        return model.eval(), EmbeddingsExtractor
    raise ValueError(
        f"No reconozco {path}: se espera models/<Problema>/<{'|'.join(NETWORKS)}>.pkl "
        f"o models/<Problema>/<{'|'.join(CLASSIC_EXTRACTORS)}>.joblib"
    )


class LoadedModel:
    """Un modelo entrenado y el extractor de features con el que se entrenó"""

    def __init__(self, path: Path):
        problem_name, name = path.parent.name, model_name(path)
        if problem_name not in PROBLEMS:
            raise ValueError(f"No reconozco el problema de {path}: se espera models/<{'|'.join(PROBLEMS)}>/...")
        problem = PROBLEMS[problem_name]
        self.problem_name = problem_name
        self.name = f"{problem_name}/{path.stem}"
        self.is_joint = name == "joint_rnn"
        self.model, extractor_class = _load(path, problem, name)
        self.extractor = _extractor(extractor_class)

    def predict(self, tokens: list[str]) -> torch.Tensor:
        """Clases predichas de cada token: (n_tokens, n_columnas). Las columnas son
        - capitalización:   [capitalización]
        - puntuación:       [inicial, final]
        - red conjunta:     [capitalización, inicial, final]
        Así la capitalización siempre es la primera columna y la puntuación las dos últimas"""
        features = self.extractor.extract(tokens)  # (n_tokens, n_features)
        if isinstance(self.model, nn.Module):
            with torch.no_grad():
                logits = self.model(features.unsqueeze(0), [len(tokens)])[0]
            pred = logits.argmax(dim=-1)
        else:
            pred = torch.as_tensor(self.model.predict(features.numpy()))
        return pred.reshape(len(tokens), -1)


def tokenize(text: str, tokenizer: BertTokenizer) -> tuple[list[str], list[list[str]]]:
    """Normaliza cada palabra como el dataset y la tokeniza. Descarta las que quedan vacías
    (signos sueltos). Devuelve las palabras normalizadas y los tokens de cada una"""
    words, word_tokens = [], []
    for word in text.split():
        normalized = strip_punctuation(word).lower()
        tokens = tokenizer.tokenize(normalized)
        if tokens:
            words.append(normalized)
            word_tokens.append(tokens)
    return words, word_tokens


def _split_by_word(pred: torch.Tensor, word_tokens: list[list[str]]) -> list[list]:
    """Agrupa las filas de `pred` (una por token) en una lista por palabra"""
    return [rows.tolist() for rows in torch.split(pred, [len(tokens) for tokens in word_tokens])]


def _all_tokens(word_tokens: list[list[str]]) -> list[str]:
    return [token for tokens in word_tokens for token in tokens]


@dataclass
class ModelSelection:
    """Los modelos elegidos: como máximo uno por problema"""
    capitalization: LoadedModel | None = None
    punctuation: LoadedModel | None = None

    def for_capitalization(self) -> LoadedModel | None:
        """El modelo de capitalización o, si no hay, la red conjunta elegida para puntuación"""
        return self.capitalization or _joint_or_none(self.punctuation)

    def for_punctuation(self) -> LoadedModel | None:
        """El modelo de puntuación o, si no hay, la red conjunta elegida para capitalización"""
        return self.punctuation or _joint_or_none(self.capitalization)


def _joint_or_none(model: LoadedModel | None) -> LoadedModel | None:
    return model if model and model.is_joint else None


def predict_capitalization(model: LoadedModel | None, word_tokens: list[list[str]]) -> Capitalizations | None:
    """Capitalización de cada palabra: la clase más votada entre sus tokens, como la métrica.
    None si no hay modelo"""
    if model is None:
        return None
    pred = model.predict(_all_tokens(word_tokens))[:, 0]  # primera columna
    return [Counter(classes).most_common(1)[0][0] for classes in _split_by_word(pred, word_tokens)]


def predict_punctuation(model: LoadedModel | None, word_tokens: list[list[str]]) -> Punctuations | None:
    """Puntuación de cada palabra: la inicial del primer token y la final del último, como la
    métrica. None si no hay modelo"""
    if model is None:
        return None
    pred = model.predict(_all_tokens(word_tokens))[:, -2:]  # dos últimas columnas: [inicial, final]
    return [(rows[0][0], rows[-1][1]) for rows in _split_by_word(pred, word_tokens)]


def capitalize(word: str, capitalization: int) -> str:
    """0 = minúscula, 1 = inicial, 2 = mixta, 3 = mayúscula. La mixta (ej: "McDonald's") no
    dice qué letras van en mayúscula, así que se marca con un * y queda en minúscula"""
    return {0: word, 1: word.capitalize(), 2: f"{word}*", 3: word.upper()}[capitalization]


def apply_capitalization(words: list[str], capitalizations: Capitalizations | None) -> list[str]:
    if capitalizations is None:
        return words
    return [capitalize(word, c) for word, c in zip(words, capitalizations)]


def apply_punctuation(words: list[str], punctuations: Punctuations | None) -> list[str]:
    if punctuations is None:
        return words
    return [INITIAL_LABELS[initial] + word + FINAL_LABELS[final] for word, (initial, final) in zip(words, punctuations)]


def format_table(words: list[str], capitalizations: Capitalizations | None, punctuations: Punctuations | None) -> str:
    """Tabla con la clase predicha de cada palabra (solo las columnas que tienen modelo)"""
    width = max(len(word) for word in words + ["palabra"]) + 2
    header = "palabra".ljust(width)
    header += "capitalización".ljust(16) if capitalizations else ""
    header += "inicial".ljust(10) + "final" if punctuations else ""
    lines = [header]
    for i, word in enumerate(words):
        line = word.ljust(width)
        line += class_name("capitalizacion", capitalizations[i]).ljust(16) if capitalizations else ""
        if punctuations:
            initial, final = punctuations[i]
            line += class_name("punt_inicial", initial).ljust(10) + class_name("punt_final", final)
        lines.append(line)
    return "\n".join(lines)


def answer(text: str, selection: ModelSelection, tokenizer: BertTokenizer, verbose: bool) -> str:
    """Normaliza el texto como el dataset y lo reconstruye: primero la capitalización y
    después la puntuación (al revés, "¿hola".capitalize() no cambiaría nada)"""
    words, word_tokens = tokenize(text, tokenizer)
    if not words:
        return "(no quedó ninguna palabra después de normalizar)"

    capitalizations = predict_capitalization(selection.for_capitalization(), word_tokens)
    punctuations = predict_punctuation(selection.for_punctuation(), word_tokens)
    output = " ".join(apply_punctuation(apply_capitalization(words, capitalizations), punctuations))

    if verbose:
        output += "\n\n" + format_table(words, capitalizations, punctuations)
    return output


# --- Menú interactivo ---

PROBLEM_LABELS = {"CapitalizationProblem": "capitalización", "PunctuationProblem": "puntuación"}
KIND_LABELS = {".pkl": "red", ".joblib": "clásico"}

ACTION_TEXT, ACTION_MODELS, ACTION_DETAIL, ACTION_CLEAR, ACTION_EXIT = "text", "models", "detail", "clear", "exit"
CLEAR_COMMANDS = {"/limpiar", "/clear"}


def available_models(problem_name: str) -> list[Path]:
    """Modelos guardados por el pipeline para ese problema que este script sabe cargar"""
    known = set(NETWORKS) | set(CLASSIC_EXTRACTORS)
    folder = Path(MODELS_DIR) / problem_name
    return sorted(
        path for path in folder.glob("*")
        if path.suffix in KIND_LABELS and model_name(path) in known
    )


def _model_choice(path: Path) -> questionary.Choice:
    """Ej: 'bidirectional_rnn   red' / 'joint_rnn   red · predice las 3 tareas'"""
    name = model_name(path)
    extra = " · predice las 3 tareas" if name == "joint_rnn" else ""
    return questionary.Choice(f"{name:<20} {run_info(path):<28} {KIND_LABELS[path.suffix]}{extra}", value=path)


def choose_models() -> dict[str, Path] | None:
    """Pide un modelo por problema (o ninguno). Devuelve {problema: path}; None si se cancela"""
    chosen = {}
    for problem_name, label in PROBLEM_LABELS.items():
        choices = [_model_choice(path) for path in available_models(problem_name)]
        if not choices:
            print(f"No hay modelos de {label} en {Path(MODELS_DIR) / problem_name}")
            continue
        path = questionary.select(
            f"Modelo de {label}:", choices=choices + [questionary.Choice("ninguno", value="")]
        ).ask()
        if path is None:  # Ctrl+C
            return None
        if path:
            chosen[problem_name] = path
    return chosen


def paths_by_problem(paths: list[Path]) -> dict[str, Path]:
    """Ordena los -m por problema (la carpeta del modelo). Falla si hay dos del mismo"""
    chosen = {}
    for path in paths:
        if path.parent.name in chosen:
            raise ValueError(f"Hay dos modelos de {path.parent.name}: {chosen[path.parent.name]} y {path}")
        chosen[path.parent.name] = path
    return chosen


def load_selection(paths: dict[str, Path]) -> ModelSelection:
    print("Cargando modelos...")
    selection = ModelSelection(
        capitalization=LoadedModel(paths["CapitalizationProblem"]) if "CapitalizationProblem" in paths else None,
        punctuation=LoadedModel(paths["PunctuationProblem"]) if "PunctuationProblem" in paths else None,
    )
    print_selection(selection)
    return selection


def print_selection(selection: ModelSelection) -> None:
    """Qué modelo predice cada problema"""
    for label, model in (("capitalización", selection.for_capitalization()),
                         ("puntuación", selection.for_punctuation())):
        print(f"  {label:<15} {'✓ ' + model.name if model else '— sin modelo'}")


def choose_action(verbose: bool) -> str:
    """Menú principal. Ctrl+C equivale a salir"""
    detail = "Ocultar detalle por palabra" if verbose else "Mostrar detalle por palabra"
    return questionary.select("¿Qué querés hacer?", choices=[
        questionary.Choice("Probar un texto", value=ACTION_TEXT),
        questionary.Choice("Cambiar modelos", value=ACTION_MODELS),
        questionary.Choice(detail, value=ACTION_DETAIL),
        questionary.Choice("Limpiar pantalla", value=ACTION_CLEAR),
        questionary.Choice("Salir", value=ACTION_EXIT),
    ]).ask() or ACTION_EXIT


def clear_screen(selection: ModelSelection) -> None:
    """Borra la terminal y vuelve a mostrar qué modelos están cargados"""
    print("\033[2J\033[3J\033[H", end="", flush=True)  # pantalla, historial y cursor arriba
    print("Modelos cargados:")
    print_selection(selection)
    print()


def try_texts(selection: ModelSelection, tokenizer: BertTokenizer, verbose: bool) -> None:
    """Pide textos hasta recibir uno vacío y muestra la reconstrucción de cada uno.
    /limpiar (o /clear) borra la pantalla"""
    while True:
        text = questionary.text("Texto (Enter vacío para volver al menú, /limpiar para borrar):").ask()
        if not text or not text.strip():
            return
        if text.strip().lower() in CLEAR_COMMANDS:
            clear_screen(selection)
            continue
        print("\n" + answer(text, selection, tokenizer, verbose) + "\n")


def run_menu(selection: ModelSelection, tokenizer: BertTokenizer, verbose: bool) -> None:
    while True:
        action = choose_action(verbose)
        if action == ACTION_EXIT:
            return
        if action == ACTION_DETAIL:
            verbose = not verbose
        elif action == ACTION_CLEAR:
            clear_screen(selection)
        elif action == ACTION_TEXT:
            try_texts(selection, tokenizer, verbose)
        elif (paths := choose_models()):
            selection = load_selection(paths)


def main() -> None:
    parser = ArgumentParser(description="Reconstruye capitalización y puntuación de un texto con modelos entrenados")
    parser.add_argument("-m", "--model", action="append", type=Path,
                        help="path a un .pkl o .joblib en models/<Problema>/ (uno por problema). "
                             "Sin -m, se eligen desde un menú")
    parser.add_argument("-v", "--verbose", action="store_true", help="mostrar la clase predicha de cada palabra")
    args = parser.parse_args()

    transformers.logging.set_verbosity_error()  # sin el reporte de carga de BERT
    transformers.logging.disable_progress_bar()
    paths = paths_by_problem(args.model) if args.model else choose_models()
    if not paths:
        print("No se eligió ningún modelo")
        return
    selection = load_selection(paths)
    tokenizer = BertTokenizer.from_pretrained(EMBEDDINGS_MODEL)
    run_menu(selection, tokenizer, args.verbose)


if __name__ == "__main__":
    main()
