"""Genera el .csv de predicciones que pide el enunciado.

Recibe el .csv de evaluación, ya tokenizado con bert-base-multilingual-cased, con una
línea por token:

    instancia_id,token_id,token

y escribe el mismo archivo con las tres columnas predichas:

    instancia_id,token_id,token,punt_inicial,punt_final,capitalización

Cada instancia se predice como una secuencia. Las palabras se reconstruyen con los "##"
(un token con "##" continúa la palabra anterior) y las predicciones se ubican como en el
dataset y en la métrica:
- capitalización: la más votada entre los tokens de la palabra, en todos sus tokens
- puntuación inicial: en el primer token de la palabra
- puntuación final: en el último token de la palabra

Se necesita un modelo por problema, o una red conjunta (que predice los dos):

    uv run python -m src.predict datasets/eval.csv -o predicciones.csv \\
        -m models/runs/CapitalizationProblem/<score>__bidirectional_rnn/<score>__bidirectional_rnn.pkl \\
        -m models/runs/PunctuationProblem/<score>__bidirectional_rnn/<score>__bidirectional_rnn.pkl
    uv run python -m src.predict datasets/eval.csv -o predicciones.csv \\
        -m models/runs/PunctuationProblem/<score>__bidirectional_joint_lstm/<score>__bidirectional_joint_lstm.pkl
"""

import csv
from argparse import ArgumentParser
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import transformers

from src.data.punctuation_dataset import FINAL_LABELS, INITIAL_LABELS
from src.inference import ModelSelection, load_selection, paths_by_problem
from src.words import first_token_mask, last_token_mask, majority_by_word, token_word_ids

INPUT_COLUMNS = ["instancia_id", "token_id", "token"]
OUTPUT_COLUMNS = INPUT_COLUMNS + ["punt_inicial", "punt_final", "capitalización"]

Row = dict[str, str]


class TokenPredictor(Protocol):
    """Lo que se necesita de un modelo cargado (ver ``src.inference.LoadedModel``)."""

    def predict(self, tokens: list[str]) -> torch.Tensor:
        """Predice las clases de cada token.

        Parameters
        ----------
        tokens : list[str]
            Tokens BERT de una instancia.

        Returns
        -------
        torch.Tensor of shape (n_tokens, n_columnas)
            La capitalización en la primera columna y la puntuación en las dos últimas.
        """
        ...


def read_instances(path: Path) -> dict[str, list[Row]]:
    """Lee el .csv de evaluación y agrupa las filas de cada instancia, en el orden del archivo.

    Parameters
    ----------
    path : Path
        .csv con las columnas ``instancia_id,token_id,token``.

    Returns
    -------
    dict[str, list[Row]]
        Las filas de cada instancia, por ``instancia_id``.

    Raises
    ------
    ValueError
        Si al archivo le faltan columnas.
    """
    instances: dict[str, list[Row]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(INPUT_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"A {path} le faltan las columnas {sorted(missing)}")
        for row in reader:
            instances.setdefault(row["instancia_id"], []).append(row)
    return instances


def capitalization_by_token(model: TokenPredictor, tokens: list[str], word_ids: np.ndarray) -> np.ndarray:
    """Predice la capitalización: la clase más votada de cada palabra, repetida en todos sus tokens.

    Parameters
    ----------
    model : TokenPredictor
        Modelo de capitalización (o red conjunta).
    tokens : list[str]
        Tokens BERT de la instancia.
    word_ids : np.ndarray of shape (n_tokens,)
        Índice de palabra de cada token.

    Returns
    -------
    np.ndarray of shape (n_tokens,)
        Clase de capitalización de cada token.
    """
    pred = model.predict(tokens)[:, 0].numpy()  # la capitalización es la primera columna
    by_word = majority_by_word(pred, word_ids)
    return by_word[word_ids]


def punctuation_by_token(model: TokenPredictor, tokens: list[str], word_ids: np.ndarray) -> tuple[list[str], list[str]]:
    """Predice la puntuación: el signo inicial en el primer token de cada palabra y el final en el último.

    Parameters
    ----------
    model : TokenPredictor
        Modelo de puntuación (o red conjunta).
    tokens : list[str]
        Tokens BERT de la instancia.
    word_ids : np.ndarray of shape (n_tokens,)
        Índice de palabra de cada token.

    Returns
    -------
    initials : list[str]
        Signo inicial de cada token (``""`` en los que no son el primero de su palabra).
    finals : list[str]
        Signo final de cada token (``""`` en los que no son el último de su palabra).
    """
    pred = model.predict(tokens)[:, -2:].numpy()  # dos últimas columnas: [inicial, final]
    first, last = first_token_mask(word_ids), last_token_mask(word_ids)
    initials = [INITIAL_LABELS[c] if is_first else "" for c, is_first in zip(pred[:, 0], first)]
    finals = [FINAL_LABELS[c] if is_last else "" for c, is_last in zip(pred[:, 1], last)]
    return initials, finals


def predict_instance(rows: list[Row], capitalization: TokenPredictor, punctuation: TokenPredictor) -> list[Row]:
    """Agrega a las filas de la instancia las tres columnas predichas.

    Parameters
    ----------
    rows : list[Row]
        Filas de la instancia, una por token.
    capitalization : TokenPredictor
        Modelo de capitalización.
    punctuation : TokenPredictor
        Modelo de puntuación.

    Returns
    -------
    list[Row]
        Las filas con ``punt_inicial``, ``punt_final`` y ``capitalización``.
    """
    tokens = [row["token"] for row in rows]
    word_ids = token_word_ids(tokens)
    caps = capitalization_by_token(capitalization, tokens, word_ids)
    initials, finals = punctuation_by_token(punctuation, tokens, word_ids)
    return [
        {
            **{column: row[column] for column in INPUT_COLUMNS},
            "punt_inicial": initial,
            "punt_final": final,
            "capitalización": str(cap),
        }
        for row, initial, final, cap in zip(rows, initials, finals, caps)
    ]


def write_predictions(path: Path, rows: list[Row]) -> None:
    """Escribe el .csv de predicciones.

    ``csv.writer`` encierra entre comillas los campos con coma, como pide el enunciado: ``","``.

    Parameters
    ----------
    path : Path
        Archivo de salida.
    rows : list[Row]
        Filas con las columnas de ``OUTPUT_COLUMNS``.
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def predict_file(input_path: Path, output_path: Path, selection: ModelSelection) -> int:
    """Predice todas las instancias del archivo y escribe el resultado.

    Parameters
    ----------
    input_path : Path
        .csv de evaluación.
    output_path : Path
        .csv de salida.
    selection : ModelSelection
        Modelos elegidos.

    Returns
    -------
    int
        Cantidad de instancias procesadas.

    Raises
    ------
    ValueError
        Si falta el modelo de algún problema (y ninguno es una red conjunta).
    """
    capitalization, punctuation = selection.for_capitalization(), selection.for_punctuation()
    if capitalization is None or punctuation is None:
        raise ValueError("Hace falta un modelo de cada problema, o una red conjunta")
    instances = read_instances(input_path)
    rows = [row for instance in instances.values() for row in predict_instance(instance, capitalization, punctuation)]
    write_predictions(output_path, rows)
    return len(instances)


def main() -> None:
    """Lee los argumentos, carga los modelos y escribe el .csv de predicciones."""
    parser = ArgumentParser(description="Predice puntuación y capitalización de un .csv tokenizado")
    parser.add_argument("input", type=Path, help="csv con las columnas instancia_id,token_id,token")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("predicciones.csv"), help="csv de salida (default: predicciones.csv)"
    )
    parser.add_argument(
        "-m",
        "--model",
        action="append",
        type=Path,
        required=True,
        help="path a un .pkl o .joblib en models/runs/<Problema>/<score>__<modelo>/ (uno por problema, o una sola red conjunta)",
    )
    args = parser.parse_args()

    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    selection = load_selection(paths_by_problem(args.model))
    n_instances = predict_file(args.input, args.output, selection)
    print(f"{n_instances} instancias predichas en {args.output}")


if __name__ == "__main__":
    main()
