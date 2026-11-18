"""Prueba interactiva de los modelos guardados por el pipeline.

Pide un texto, lo normaliza igual que el dataset (minúsculas y sin puntuación), corre los
modelos y reconstruye el texto con la capitalización y la puntuación predichas.

Sin argumentos abre un menú para elegir los modelos guardados en
models/runs/<Problema>/<score>__<modelo>/ (problema → modelo) y probar textos:

    uv run python -m src.inference

Acepta redes (.pkl, solo los pesos) y modelos clásicos (.joblib, un ClassicModel: el
RandomForest entero y, si lo usa, el diccionario de capitalización).
Cada modelo predice solo su problema; para reconstruir todo hay que elegir uno de cada
problema, o la red conjunta, que predice los dos. Los modelos también se
pueden pasar directamente, salteando el menú de modelos:

    uv run python -m src.inference -m models/runs/CapitalizationProblem/0.706__bidirectional_rnn/0.706__bidirectional_rnn.pkl \\
                                   -m models/runs/PunctuationProblem/0.480__classic_embeddings/0.480__classic_embeddings.joblib
    uv run python -m src.inference -m models/runs/PunctuationProblem/0.792__joint_conv_bilstm/0.792__joint_conv_bilstm.pkl -v

Se usa como máximo un modelo por problema. Si falta el de un problema y el otro es la red
conjunta, la red conjunta predice los dos.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import questionary
import torch
import transformers
from torch import nn
from transformers import BertTokenizer

from src.constants import EMBEDDINGS_MODEL, RUN_SEPARATOR, RUNS_DIR
from src.data import normalize_word
from src.data.capitalization_dataset import LOWERCASE, MIXED, TITLE, UPPERCASE
from src.data.punctuation_dataset import FINAL_LABELS, INITIAL_LABELS
from src.features import (
    AverageEmbeddingsExtractor,
    ClassicFeatureExtractor,
    EmbeddingsExtractor,
    TextFeatureExtractor,
)
from src.problem.joint import (
    create_joint_lstm_model,
    create_joint_rnn_model,
    create_conv_bilstm_model,
    create_projected_bigru_model,
    create_residual_stacked_bilstm_model,
)
from src.metrics.base import class_name
from src.problem import CapitalizationProblem, Problem, PunctuationProblem
from src.training.neural import predict
from src.training.report import load_metadata
from src.words import first_token_mask, last_token_mask, majority_by_word, token_word_ids

PROBLEMS: dict[str, Problem] = {
    "CapitalizationProblem": CapitalizationProblem(),
    "PunctuationProblem": PunctuationProblem(),
}

# Mismos nombres y extractores que en build_experiments: el archivo se guarda con ese nombre
NETWORKS: dict[str, Callable[[Problem], nn.Module]] = {
    "standard_rnn": lambda problem: problem.standard_rnn_model(),
    "bidirectional_rnn": lambda problem: problem.bidirectional_rnn_model(),
    "joint_rnn": lambda _: create_joint_rnn_model(),
    "bidirectional_joint_rnn": lambda _: create_joint_rnn_model(bidirectional=True),
    "joint_lstm": lambda _: create_joint_lstm_model(),
    "bidirectional_joint_lstm": lambda _: create_joint_lstm_model(bidirectional=True),
    "joint_projected_bigru": lambda _: create_projected_bigru_model(),
    "joint_residual_stacked_bilstm": lambda _: create_residual_stacked_bilstm_model(),
    "joint_conv_bilstm": lambda _: create_conv_bilstm_model(),
}
# Redes que predicen las 3 tareas a la vez
JOINT_NETWORKS = {
    "joint_rnn",
    "bidirectional_joint_rnn",
    "joint_lstm",
    "bidirectional_joint_lstm",
    "joint_projected_bigru",
    "joint_residual_stacked_bilstm",
    "joint_conv_bilstm",
}
CLASSIC_EXTRACTORS: dict[str, type[TextFeatureExtractor]] = {
    "classic_features": ClassicFeatureExtractor,
    "classic_features_dictionary": ClassicFeatureExtractor,
    "classic_embeddings": AverageEmbeddingsExtractor,
}

Capitalizations = list[int]
"""Clase de capitalización de cada palabra (LOWERCASE, TITLE, MIXED o UPPERCASE)"""

Punctuations = list[tuple[int, int]]
"""(puntuación inicial, puntuación final) de cada palabra: índices de INITIAL_LABELS y FINAL_LABELS"""


def model_name(path: Path) -> str:
    """Extrae el nombre del modelo, sin el score.

    Parameters
    ----------
    path : Path
        Archivo ``models/runs/<Problema>/<score>__<modelo>/<score>__<modelo>.<ext>``.

    Returns
    -------
    str
        Ej: ``'0.812__standard_rnn.pkl'`` -> ``'standard_rnn'``.
    """
    return path.stem.split(RUN_SEPARATOR)[-1]


def run_info(path: Path) -> str:
    """Extrae el score del modelo de su nombre de archivo.

    Parameters
    ----------
    path : Path
        Archivo ``models/runs/<Problema>/<score>__<modelo>/<score>__<modelo>.<ext>``.

    Returns
    -------
    str
        Ej: ``'F1 0.812'``.
    """
    score, _ = path.stem.split(RUN_SEPARATOR)
    return f"F1 {score}"


def problem_of(path: Path) -> str:
    """Devuelve el problema del modelo: la carpeta de arriba de la del modelo.

    Parameters
    ----------
    path : Path
        Archivo ``models/runs/<Problema>/<score>__<modelo>/<score>__<modelo>.<ext>``.

    Returns
    -------
    str
        Ej: ``'CapitalizationProblem'``.
    """
    return path.parent.parent.name


def run_of(path: Path) -> str:
    """Devuelve la corrida del modelo, según la metadata.json de su carpeta.

    Parameters
    ----------
    path : Path
        Archivo ``models/runs/<Problema>/<score>__<modelo>/<score>__<modelo>.<ext>``.

    Returns
    -------
    str
        Ej: ``'2026-11-12__19-30-05'``; ``'?'`` si la carpeta no tiene metadata.
    """
    try:
        return load_metadata(path.parent)["run"]
    except FileNotFoundError:
        return "?"


@cache
def _extractor(extractor_class: type[TextFeatureExtractor]) -> TextFeatureExtractor:
    """Crea una sola instancia por tipo de extractor (los de embeddings cargan BERT).

    Parameters
    ----------
    extractor_class : type[TextFeatureExtractor]
        Clase del extractor.

    Returns
    -------
    TextFeatureExtractor
        La instancia, compartida entre llamadas.
    """
    return extractor_class()


def head_dim_of(state: dict[str, torch.Tensor]) -> int | None:
    """Deduce el tamaño de la capa oculta de la cabeza de una BiRNN a partir de sus pesos.

    Parameters
    ----------
    state : dict[str, torch.Tensor]
        Pesos guardados (``state_dict``).

    Returns
    -------
    int or None
        Filas de la primera capa de la cabeza no lineal, o ``None`` si la cabeza es lineal
        (en ese caso los pesos son ``out.weight``, no ``out.0.weight``).
    """
    for key in ("out.0.weight", "out_ini.0.weight"):
        if key in state:
            return state[key].shape[0]
    return None


def create_network(name: str, problem: Problem, state: dict[str, torch.Tensor]) -> nn.Module:
    """Crea la red sin entrenar con la misma arquitectura que los pesos guardados.

    Parameters
    ----------
    name : str
        Nombre del modelo (ver ``NETWORKS``).
    problem : Problem
        Problema del modelo.
    state : dict[str, torch.Tensor]
        Pesos guardados, para deducir la cabeza de la BiRNN (lineal o no lineal).

    Returns
    -------
    torch.nn.Module
        La red, lista para ``load_state_dict``.
    """
    if name == "bidirectional_rnn":
        return problem.bidirectional_rnn_model(head_dim=head_dim_of(state))
    return NETWORKS[name](problem)


def load_model(path: Path, problem: Problem, name: str) -> tuple[Any, type[TextFeatureExtractor]]:
    """Carga un modelo guardado y elige el extractor con el que se entrenó.

    Parameters
    ----------
    path : Path
        Archivo ``models/runs/<Problema>/<score>__<modelo>/<score>__<modelo>.<ext>``.
    problem : Problem
        Problema del modelo, para crear la red antes de cargar los pesos.
    name : str
        Nombre del modelo (ver ``model_name``).

    Returns
    -------
    model : Any
        La red en modo eval o el ``ClassicModel``.
    extractor_class : type[TextFeatureExtractor]
        Clase del extractor de features del modelo.

    Raises
    ------
    ValueError
        Si la extensión o el nombre no corresponden a un modelo conocido.
    """
    if path.suffix == ".joblib" and name in CLASSIC_EXTRACTORS:
        return joblib.load(path), CLASSIC_EXTRACTORS[name]
    if path.suffix == ".pkl" and name in NETWORKS:
        state = torch.load(path, weights_only=True, map_location="cpu")
        model = create_network(name, problem, state)
        model.load_state_dict(state)
        return model.eval(), EmbeddingsExtractor
    raise ValueError(
        f"No reconozco {path}: se espera models/runs/<Problema>/<score>__<modelo>/<score>__<{'|'.join(NETWORKS)}>.pkl "
        f"o .../<score>__<{'|'.join(CLASSIC_EXTRACTORS)}>.joblib"
    )


class LoadedModel:
    """Un modelo entrenado y el extractor de features con el que se entrenó.

    Parameters
    ----------
    path : Path
        Archivo ``models/runs/<Problema>/<score>__<modelo>/<score>__<modelo>.<ext>``.

    Attributes
    ----------
    name : str
        ``<ejecución>/<Problema>/<score>__<modelo>``, para mostrar.
    is_joint : bool
        Si es una red conjunta (predice los dos problemas).
    model : Any
        La red o el ``ClassicModel``.
    extractor : TextFeatureExtractor
        Extractor de features del modelo.

    Raises
    ------
    ValueError
        Si la carpeta del archivo no es un problema conocido.
    """

    def __init__(self, path: Path) -> None:
        problem_name, name = problem_of(path), model_name(path)
        if problem_name not in PROBLEMS:
            raise ValueError(
                f"No reconozco el problema de {path}: se espera models/runs/<{'|'.join(PROBLEMS)}>/<score>__<modelo>/..."
            )
        problem = PROBLEMS[problem_name]
        self.name = f"{problem_name}/{path.stem} ({run_of(path)})"
        self.is_joint = name in JOINT_NETWORKS
        self.model, extractor_class = load_model(path, problem, name)
        self.extractor = _extractor(extractor_class)

    def predict(self, tokens: list[str]) -> torch.Tensor:
        """Predice las clases de cada token.

        Los modelos clásicos predicen por palabra: cada token recibe la clase de su palabra.

        Parameters
        ----------
        tokens : list[str]
            Tokens BERT de una instancia.

        Returns
        -------
        torch.Tensor of shape (n_tokens, n_columnas)
            Las columnas son ``[capitalización]`` (capitalización), ``[inicial, final]``
            (puntuación) o ``[capitalización, inicial, final]`` (red conjunta). Así la
            capitalización siempre es la primera columna y la puntuación las dos últimas.
        """
        if isinstance(self.model, nn.Module):
            features = self.extractor.extract(tokens)  # (n_tokens, n_features)
            pred = predict(self.model, features.unsqueeze(0), [len(tokens)])[0]
        else:
            pred = torch.as_tensor(self.model.predict_words(self.extractor, tokens))[token_word_ids(tokens)]
        return pred.reshape(len(tokens), -1)


def tokenize(text: str, tokenizer: BertTokenizer) -> tuple[list[str], list[list[str]]]:
    """Normaliza cada palabra como el dataset y la tokeniza.

    Descarta las que quedan vacías (signos sueltos).

    Parameters
    ----------
    text : str
        Texto ingresado.
    tokenizer : BertTokenizer
        Tokenizador de ``EMBEDDINGS_MODEL``.

    Returns
    -------
    words : list[str]
        Las palabras normalizadas.
    word_tokens : list[list[str]]
        Los tokens de cada palabra.
    """
    words, word_tokens = [], []
    for word in text.split():
        normalized = normalize_word(word)
        tokens = tokenizer.tokenize(normalized)
        if tokens:
            words.append(normalized)
            word_tokens.append(tokens)
    return words, word_tokens


def _word_ids(word_tokens: list[list[str]]) -> np.ndarray:
    """Calcula el índice de la palabra de cada token.

    Parameters
    ----------
    word_tokens : list[list[str]]
        Tokens de cada palabra.

    Returns
    -------
    np.ndarray of shape (n_tokens,)
        Ej: ``[["hol", "##a"], ["la"]]`` -> ``[0, 0, 1]``.
    """
    return np.repeat(np.arange(len(word_tokens)), [len(tokens) for tokens in word_tokens])


def _all_tokens(word_tokens: list[list[str]]) -> list[str]:
    """Aplana los tokens de todas las palabras.

    Parameters
    ----------
    word_tokens : list[list[str]]
        Tokens de cada palabra.

    Returns
    -------
    list[str]
        Todos los tokens, en orden.
    """
    return [token for tokens in word_tokens for token in tokens]


@dataclass
class ModelSelection:
    """Los modelos elegidos: como máximo uno por problema.

    Attributes
    ----------
    capitalization : LoadedModel or None
        Modelo elegido para capitalización.
    punctuation : LoadedModel or None
        Modelo elegido para puntuación.
    """

    capitalization: LoadedModel | None = None
    punctuation: LoadedModel | None = None

    def for_capitalization(self) -> LoadedModel | None:
        """Elige el modelo de capitalización o, si no hay, la red conjunta elegida para puntuación.

        Returns
        -------
        LoadedModel or None
            El modelo que predice la capitalización, o ``None``.
        """
        return self.capitalization or _joint_or_none(self.punctuation)

    def for_punctuation(self) -> LoadedModel | None:
        """Elige el modelo de puntuación o, si no hay, la red conjunta elegida para capitalización.

        Returns
        -------
        LoadedModel or None
            El modelo que predice la puntuación, o ``None``.
        """
        return self.punctuation or _joint_or_none(self.capitalization)


def _joint_or_none(model: LoadedModel | None) -> LoadedModel | None:
    """Devuelve el modelo solo si es una red conjunta.

    Parameters
    ----------
    model : LoadedModel or None
        Modelo elegido para el otro problema.

    Returns
    -------
    LoadedModel or None
        ``model`` si es conjunta, si no ``None``.
    """
    return model if model and model.is_joint else None


def predict_capitalization(model: LoadedModel | None, word_tokens: list[list[str]]) -> Capitalizations | None:
    """Predice la capitalización de cada palabra: la clase más votada entre sus tokens, como la métrica.

    Parameters
    ----------
    model : LoadedModel or None
        Modelo que predice la tarea, o ``None``.
    word_tokens : list[list[str]]
        Tokens de cada palabra normalizada.

    Returns
    -------
    Capitalizations or None
        La clase de cada palabra, o ``None`` si no hay modelo.
    """
    if model is None:
        return None
    pred = model.predict(_all_tokens(word_tokens))[:, 0].numpy()  # primera columna
    return majority_by_word(pred, _word_ids(word_tokens)).tolist()


def predict_punctuation(model: LoadedModel | None, word_tokens: list[list[str]]) -> Punctuations | None:
    """Predice la puntuación de cada palabra: la inicial del primer token y la final del último, como la métrica.

    Parameters
    ----------
    model : LoadedModel or None
        Modelo que predice la tarea, o ``None``.
    word_tokens : list[list[str]]
        Tokens de cada palabra normalizada.

    Returns
    -------
    Punctuations or None
        ``(inicial, final)`` de cada palabra, o ``None`` si no hay modelo.
    """
    if model is None:
        return None
    pred = model.predict(_all_tokens(word_tokens))[:, -2:].numpy()  # dos últimas columnas: [inicial, final]
    word_ids = _word_ids(word_tokens)
    initials = pred[first_token_mask(word_ids), 0].tolist()
    finals = pred[last_token_mask(word_ids), 1].tolist()
    return list(zip(initials, finals))


def capitalize(word: str, capitalization: int) -> str:
    """Aplica la clase de capitalización a la palabra.

    La mixta (ej: "McDonald's") no dice qué letras van en mayúscula, así que se marca
    con un * y queda en minúscula.

    Parameters
    ----------
    word : str
        Palabra normalizada.
    capitalization : int
        ``LOWERCASE``, ``TITLE``, ``MIXED`` o ``UPPERCASE``.

    Returns
    -------
    str
        La palabra capitalizada.
    """
    return {
        LOWERCASE: word,
        TITLE: word.capitalize(),
        MIXED: f"{word}*",
        UPPERCASE: word.upper(),
    }[capitalization]


def apply_capitalization(words: list[str], capitalizations: Capitalizations | None) -> list[str]:
    """Aplica la capitalización predicha a cada palabra.

    Parameters
    ----------
    words : list[str]
        Palabras normalizadas.
    capitalizations : Capitalizations or None
        Clase de cada palabra; con ``None`` las palabras quedan igual.

    Returns
    -------
    list[str]
        Las palabras capitalizadas.
    """
    if capitalizations is None:
        return words
    return [capitalize(word, c) for word, c in zip(words, capitalizations)]


def apply_punctuation(words: list[str], punctuations: Punctuations | None) -> list[str]:
    """Agrega a cada palabra sus signos inicial y final predichos.

    Parameters
    ----------
    words : list[str]
        Palabras (ya capitalizadas).
    punctuations : Punctuations or None
        ``(inicial, final)`` de cada palabra; con ``None`` las palabras quedan igual.

    Returns
    -------
    list[str]
        Las palabras con su puntuación.
    """
    if punctuations is None:
        return words
    return [INITIAL_LABELS[initial] + word + FINAL_LABELS[final] for word, (initial, final) in zip(words, punctuations)]


def format_table(words: list[str], capitalizations: Capitalizations | None, punctuations: Punctuations | None) -> str:
    """Arma una tabla con la clase predicha de cada palabra (solo las columnas que tienen modelo).

    Parameters
    ----------
    words : list[str]
        Palabras normalizadas.
    capitalizations : Capitalizations or None
        Clase de capitalización de cada palabra.
    punctuations : Punctuations or None
        ``(inicial, final)`` de cada palabra.

    Returns
    -------
    str
        La tabla, una línea por palabra.
    """
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
    """Normaliza el texto como el dataset y lo reconstruye con los modelos elegidos.

    Aplica primero la capitalización y después la puntuación (al revés,
    ``"¿hola".capitalize()`` no cambiaría nada).

    Parameters
    ----------
    text : str
        Texto ingresado.
    selection : ModelSelection
        Modelos elegidos.
    tokenizer : BertTokenizer
        Tokenizador de ``EMBEDDINGS_MODEL``.
    verbose : bool
        Si agrega la tabla con la clase de cada palabra.

    Returns
    -------
    str
        El texto reconstruido (y la tabla, si ``verbose``).
    """
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
    """Lista los modelos de ese problema que este script sabe cargar.

    Busca en ``models/runs/<Problema>/<score>__<modelo>/``.

    Parameters
    ----------
    problem_name : str
        Nombre del problema.

    Returns
    -------
    list[Path]
        Los archivos de los modelos, el de mejor score primero.
    """
    known = set(NETWORKS) | set(CLASSIC_EXTRACTORS)
    return sorted(
        (
            path
            for path in (Path(RUNS_DIR) / problem_name).glob("*/*")
            if path.suffix in KIND_LABELS and path.stem.count(RUN_SEPARATOR) == 1 and model_name(path) in known
        ),
        key=lambda path: path.name,
        reverse=True,
    )


def _model_choice(path: Path) -> questionary.Choice:
    """Arma la opción del menú de un modelo.

    Parameters
    ----------
    path : Path
        Archivo ``models/runs/<Problema>/<score>__<modelo>/<score>__<modelo>.<ext>``.

    Returns
    -------
    questionary.Choice
        Ej: ``'bidirectional_rnn    F1 0.871   red     2026-11-12__19-30-05'`` (las conjuntas
        agregan ``· predice las 3 tareas``).
    """
    name = model_name(path)
    extra = " · predice las 3 tareas" if name in JOINT_NETWORKS else ""
    kind = KIND_LABELS[path.suffix]
    return questionary.Choice(f"{name:<28} {run_info(path):<10} {kind:<8} {run_of(path)}{extra}", value=path)


def choose_model(problem_name: str, label: str) -> Path | str | None:
    """Pide el modelo del problema, de mejor a peor score.

    Parameters
    ----------
    problem_name : str
        Nombre del problema.
    label : str
        Nombre del problema para mostrar.

    Returns
    -------
    Path or str or None
        El path elegido, ``''`` si no se elige ninguno y ``None`` si se cancela.
    """
    none = questionary.Choice("ninguno", value="")
    choices = [_model_choice(path) for path in available_models(problem_name)] + [none]
    return questionary.select(f"Modelo de {label}:", choices=choices).ask()


def choose_models() -> dict[str, Path] | None:
    """Pide un modelo por problema (o ninguno).

    Returns
    -------
    dict[str, Path] or None
        ``{problema: path}``; ``None`` si se cancela.
    """
    chosen = {}
    for problem_name, label in PROBLEM_LABELS.items():
        if not available_models(problem_name):
            print(f"No hay modelos de {label} en {Path(RUNS_DIR) / problem_name}/")
            continue
        path = choose_model(problem_name, label)
        if path is None:  # Ctrl+C
            return None
        if path:
            chosen[problem_name] = path
    return chosen


def paths_by_problem(paths: list[Path]) -> dict[str, Path]:
    """Ordena los ``-m`` por problema (la carpeta del archivo).

    Parameters
    ----------
    paths : list[Path]
        Paths pasados con ``-m``.

    Returns
    -------
    dict[str, Path]
        ``{problema: path}``.

    Raises
    ------
    ValueError
        Si hay dos modelos del mismo problema.
    """
    chosen = {}
    for path in paths:
        problem_name = problem_of(path)
        if problem_name in chosen:
            raise ValueError(f"Hay dos modelos de {problem_name}: {chosen[problem_name]} y {path}")
        chosen[problem_name] = path
    return chosen


def load_selection(paths: dict[str, Path]) -> ModelSelection:
    """Carga los modelos elegidos y muestra qué modelo predice cada problema.

    Parameters
    ----------
    paths : dict[str, Path]
        ``{problema: path}``.

    Returns
    -------
    ModelSelection
        Los modelos cargados.
    """
    print("Cargando modelos...")
    selection = ModelSelection(
        capitalization=LoadedModel(paths["CapitalizationProblem"]) if "CapitalizationProblem" in paths else None,
        punctuation=LoadedModel(paths["PunctuationProblem"]) if "PunctuationProblem" in paths else None,
    )
    print_selection(selection)
    return selection


def print_selection(selection: ModelSelection) -> None:
    """Muestra qué modelo predice cada problema.

    Parameters
    ----------
    selection : ModelSelection
        Modelos elegidos.
    """
    for label, model in (
        ("capitalización", selection.for_capitalization()),
        ("puntuación", selection.for_punctuation()),
    ):
        print(f"  {label:<15} {'✓ ' + model.name if model else '— sin modelo'}")


def choose_action(verbose: bool) -> str:
    """Muestra el menú principal. Ctrl+C equivale a salir.

    Parameters
    ----------
    verbose : bool
        Si el detalle por palabra está activo (cambia el texto de esa opción).

    Returns
    -------
    str
        Una de las constantes ``ACTION_*``.
    """
    detail = "Ocultar detalle por palabra" if verbose else "Mostrar detalle por palabra"
    return (
        questionary.select(
            "¿Qué querés hacer?",
            choices=[
                questionary.Choice("Probar un texto", value=ACTION_TEXT),
                questionary.Choice("Cambiar modelos", value=ACTION_MODELS),
                questionary.Choice(detail, value=ACTION_DETAIL),
                questionary.Choice("Limpiar pantalla", value=ACTION_CLEAR),
                questionary.Choice("Salir", value=ACTION_EXIT),
            ],
        ).ask()
        or ACTION_EXIT
    )


def clear_screen(selection: ModelSelection) -> None:
    """Borra la terminal y vuelve a mostrar qué modelos están cargados.

    Parameters
    ----------
    selection : ModelSelection
        Modelos cargados.
    """
    print("\033[2J\033[3J\033[H", end="", flush=True)  # pantalla, historial y cursor arriba
    print("Modelos cargados:")
    print_selection(selection)
    print()


def try_texts(selection: ModelSelection, tokenizer: BertTokenizer, verbose: bool) -> None:
    """Pide textos hasta recibir uno vacío y muestra la reconstrucción de cada uno.

    ``/limpiar`` (o ``/clear``) borra la pantalla.

    Parameters
    ----------
    selection : ModelSelection
        Modelos cargados.
    tokenizer : BertTokenizer
        Tokenizador de ``EMBEDDINGS_MODEL``.
    verbose : bool
        Si muestra la tabla con la clase de cada palabra.
    """
    while True:
        text = questionary.text("Texto (Enter vacío para volver al menú, /limpiar para borrar):").ask()
        if not text or not text.strip():
            return
        if text.strip().lower() in CLEAR_COMMANDS:
            clear_screen(selection)
            continue
        print("\n" + answer(text, selection, tokenizer, verbose) + "\n")


def run_menu(selection: ModelSelection, tokenizer: BertTokenizer, verbose: bool) -> None:
    """Corre el menú principal hasta que se elige salir.

    Parameters
    ----------
    selection : ModelSelection
        Modelos cargados al empezar.
    tokenizer : BertTokenizer
        Tokenizador de ``EMBEDDINGS_MODEL``.
    verbose : bool
        Si arranca mostrando el detalle por palabra.
    """
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
        elif paths := choose_models():
            selection = load_selection(paths)


def main() -> None:
    """Lee los argumentos, carga los modelos (de ``-m`` o del menú) y abre el menú principal."""
    parser = ArgumentParser(description="Reconstruye capitalización y puntuación de un texto con modelos entrenados")
    parser.add_argument(
        "-m",
        "--model",
        action="append",
        type=Path,
        help="path a un .pkl o .joblib en models/runs/<Problema>/<score>__<modelo>/ (uno por problema). "
        "Sin -m, se eligen desde un menú",
    )
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
