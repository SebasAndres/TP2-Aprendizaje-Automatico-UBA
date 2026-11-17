"""Atributos "a la antigua" (sin embeddings) para los modelos clásicos, a nivel palabra.

Cada palabra se describe con:
- sus propios atributos: largo, vocales, consonantes, acentos, cantidad de tokens, si es una
  palabra interrogativa, si tiene dígitos o signos (ej: "mcdonald's")
- su posición en la instancia: primera, última y posición relativa
- los atributos propios de las CONTEXT_WORDS palabras anteriores y siguientes (ceros si no
  existen). El modelo clásico no procesa secuencias, pero así cada fila trae algo de contexto:
  ej. si la siguiente palabra es "qué", es probable que esta lleve una coma o un punto
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.features.base import TextFeatureExtractor
from src.words import split_words, token_word_ids

VOCALES = set("aeiouáéíóúü")
ACENTOS = set("áéíóú")
# Con tilde solo en preguntas y exclamaciones, y el texto normalizado conserva las tildes
INTERROGATIVAS = {
    "qué",
    "quién",
    "quiénes",
    "cuál",
    "cuáles",
    "cuándo",
    "cuánto",
    "cuánta",
    "cuántos",
    "cuántas",
    "dónde",
    "adónde",
    "cómo",
}
# Cuántas palabras de cada lado se agregan como contexto
CONTEXT_WORDS = 2


def _word_features(word: str, n_tokens: int) -> list[float]:
    """Calcula los atributos propios de la palabra según su texto.

    Parameters
    ----------
    word : str
        Palabra normalizada (tokens unidos, ver ``split_words``).
    n_tokens : int
        Cantidad de tokens BERT de la palabra.

    Returns
    -------
    list[float]
        Largo, vocales, consonantes, proporción de vocales, acentos, cantidad de
        tokens, si es interrogativa, si tiene dígitos y si tiene signos o dígitos.
    """
    length = len(word)
    num_vowels = sum(char in VOCALES for char in word)
    return [
        float(length),
        float(num_vowels),
        float(length - num_vowels),  # consonantes (y otros caracteres)
        num_vowels / length if length else 0.0,  # proporción de vocales
        float(sum(char in ACENTOS for char in word)),  # acentos
        float(n_tokens),  # palabras raras se parten en más tokens
        float(word in INTERROGATIVAS),
        float(any(char.isdigit() for char in word)),
        float(not word.isalpha()),  # tiene signos o dígitos: "mcdonald's", "2x1"
    ]


def _position_features(n_words: int) -> torch.Tensor:
    """Calcula los atributos de posición de cada palabra en la instancia.

    Parameters
    ----------
    n_words : int
        Cantidad de palabras de la instancia.

    Returns
    -------
    torch.Tensor of shape (n_words, 3)
        Es la primera, es la última y posición relativa entre 0 y 1.
    """
    position = torch.arange(n_words, dtype=torch.float32)
    return torch.stack([position == 0, position == n_words - 1, position / max(n_words - 1, 1)], dim=1).float()


def _context_features(own: torch.Tensor, context: int) -> torch.Tensor:
    """Junta los atributos de las ``context`` palabras anteriores y siguientes de cada palabra.

    Las posiciones fuera de la instancia se completan con ceros.

    Parameters
    ----------
    own : torch.Tensor of shape (n_palabras, n_atributos)
        Atributos propios de cada palabra.
    context : int
        Cuántas palabras de cada lado se agregan.

    Returns
    -------
    torch.Tensor of shape (n_palabras, 2 * context * n_atributos)
        Los atributos de los vecinos, de la palabra más lejana a la izquierda a la más
        lejana a la derecha.
    """
    n_words = own.size(0)
    padded = F.pad(own, (0, 0, context, context))  # filas de ceros arriba y abajo
    offsets = [offset for offset in range(-context, context + 1) if offset != 0]
    return torch.cat([padded[context + offset : context + offset + n_words] for offset in offsets], dim=1)


class ClassicFeatureExtractor(TextFeatureExtractor):
    """Feature extractor sin embeddings: atributos de la palabra, su posición y su contexto."""

    def extract_words(self, tokens: list[str]) -> torch.Tensor:
        """Calcula las features de cada palabra.

        Parameters
        ----------
        tokens : list[str]
            Tokens BERT de una instancia.

        Returns
        -------
        torch.Tensor of shape (n_palabras, n_features)
            Atributos propios, de posición y de las ``CONTEXT_WORDS`` palabras de cada lado.
        """
        n_tokens = np.bincount(token_word_ids(tokens))
        own = torch.tensor(
            [_word_features(word, n) for word, n in zip(split_words(tokens), n_tokens)], dtype=torch.float32
        )
        return torch.cat([own, _position_features(own.size(0)), _context_features(own, CONTEXT_WORDS)], dim=1)

    def extract(self, tokens: list[str]) -> torch.Tensor:
        """Calcula las features de cada token: las de su palabra.

        Parameters
        ----------
        tokens : list[str]
            Tokens BERT de una instancia.

        Returns
        -------
        torch.Tensor of shape (n_tokens, n_features)
            Cada token con los atributos de su palabra.
        """
        return self.extract_words(tokens)[torch.from_numpy(token_word_ids(tokens))]
