"""Convierte documentos en instancias de entrenamiento: listas de palabras con su puntuación
y capitalización originales.

Hay dos formas de cortar:
- por oración: cada instancia es una oración. Casi siempre empieza con mayúscula y termina
  con un signo, así que un modelo puede aprender "posición = etiqueta".
- por ventana: cada instancia son palabras seguidas del documento, sin solapamiento, así
  que pueden empezar y terminar en cualquier parte de una oración. Como no se solapan, dos
  ventanas nunca comparten texto y el split train/test no filtra palabras de un lado al otro.
  El largo de cada ventana se sortea entre un mínimo y un máximo, para que el modelo vea
  entradas cortas y largas como las que puede recibir al predecir.
"""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.constants import RANDOM_SEED, WINDOW_DISTRIBUTION, WINDOW_MAX_SIZE, WINDOW_MIN_SIZE
from src.data.sources import Document

Sample = list[str]  # palabras de una instancia
Distribution = Literal["uniform", "normal"]
DISTRIBUTIONS: tuple[Distribution, ...] = ("uniform", "normal")


@dataclass(frozen=True)
class WindowSpec:
    """Cómo se sortea el largo (en palabras) de cada ventana:
    - uniform: cualquier largo entre min_size y max_size con la misma probabilidad
    - normal: centrada en el punto medio, con desvío (max - min) / 6, recortada a [min, max]
    Con min_size == max_size todas las ventanas tienen el mismo largo."""

    min_size: int = WINDOW_MIN_SIZE
    max_size: int = WINDOW_MAX_SIZE
    distribution: Distribution = WINDOW_DISTRIBUTION

    def __post_init__(self):
        if not 1 <= self.min_size <= self.max_size:
            raise ValueError(f"Se necesita 1 <= min_size <= max_size, no {self.min_size}, {self.max_size}")
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(f"Distribución desconocida {self.distribution!r}, opciones: {DISTRIBUTIONS}")

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Sortea `n` largos de ventana"""
        if self.distribution == "uniform":
            return rng.integers(self.min_size, self.max_size + 1, size=n)
        mean = (self.min_size + self.max_size) / 2
        std = (self.max_size - self.min_size) / 6
        return np.clip(np.rint(rng.normal(mean, std, size=n)), self.min_size, self.max_size).astype(int)


def by_sentence(documents: list[Document]) -> list[Sample]:
    """Una instancia por oración"""
    return [sentence.split() for document in documents for sentence in document]


def by_window(documents: list[Document], window: WindowSpec, seed: int = RANDOM_SEED) -> list[Sample]:
    """Ventanas de palabras seguidas con largo sorteado según `window`, sin solaparse. Las
    ventanas no cruzan de un documento a otro; la última de cada documento puede ser más
    corta que el mínimo. Con la misma semilla los cortes son siempre los mismos, así los
    splits train/test/control no cambian entre corridas"""
    rng = np.random.default_rng(seed)
    samples = []
    for document in documents:
        words = [word for sentence in document for word in sentence.split()]
        # Alcanzan len(words) // min_size + 1 largos para cubrir el documento entero
        lengths = window.draw(rng, len(words) // window.min_size + 1)
        starts = np.concatenate([[0], np.cumsum(lengths)])
        starts = starts[starts < len(words)]
        ends = np.append(starts[1:], len(words))
        samples.extend(words[start:end] for start, end in zip(starts, ends))
    return samples


def build_samples(documents: list[Document], window: WindowSpec | None) -> list[Sample]:
    """Ventanas según `window`, o una instancia por oración si es None"""
    return by_sentence(documents) if window is None else by_window(documents, window)


def add_window_args(parser: ArgumentParser) -> None:
    """Agrega a un CLI las opciones para elegir cómo cortar las instancias"""
    parser.add_argument(
        "--window-size",
        type=int,
        nargs="+",
        metavar="N",
        default=[WINDOW_MIN_SIZE, WINDOW_MAX_SIZE],
        help="largo de las ventanas en palabras: MIN MAX para sortearlo, un solo N para un largo fijo, "
        f"0 para una oración por instancia (default: {WINDOW_MIN_SIZE} {WINDOW_MAX_SIZE})",
    )
    parser.add_argument(
        "--window-distribution",
        choices=DISTRIBUTIONS,
        default=WINDOW_DISTRIBUTION,
        help=f"cómo se sortea el largo de cada ventana (default: {WINDOW_DISTRIBUTION})",
    )


def window_from_args(args: Namespace) -> WindowSpec | None:
    """El WindowSpec elegido con las opciones de add_window_args (None = por oración)"""
    sizes = args.window_size
    if sizes == [0]:
        return None
    if len(sizes) > 2:
        raise ValueError(f"--window-size recibe 1 o 2 valores, no {len(sizes)}")
    return WindowSpec(sizes[0], sizes[-1], args.window_distribution)
