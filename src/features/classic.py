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
from src.words import token_word_ids

VOCALES = set("aeiouáéíóúü")
ACENTOS = set("áéíóú")
# Con tilde solo en preguntas y exclamaciones, y el texto normalizado conserva las tildes
INTERROGATIVAS = {
    "qué", "quién", "quiénes", "cuál", "cuáles", "cuándo", "cuánto", "cuánta", "cuántos",
    "cuántas", "dónde", "adónde", "cómo",
}
# Cuántas palabras de cada lado se agregan como contexto
CONTEXT_WORDS = 2


def split_words(tokens: list[str]) -> list[str]:
    """Junta los tokens de cada palabra (un "##" continúa la anterior), con el mismo criterio
    que token_word_ids. Ej: ["cu", "##ánd", "##o", "va"] -> ["cuándo", "va"]"""
    words: list[str] = []
    for i, token in enumerate(tokens):
        if i > 0 and token.startswith("##"):
            words[-1] += token.removeprefix("##")
        else:
            words.append(token.removeprefix("##"))
    return words


def _word_features(word: str, n_tokens: int) -> list[float]:
    """Atributos de la palabra según su texto"""
    length = len(word)
    num_vowels = sum(char in VOCALES for char in word)
    return [
        float(length),
        float(num_vowels),
        float(length - num_vowels),                    # consonantes (y otros caracteres)
        num_vowels / length if length else 0.0,        # proporción de vocales
        float(sum(char in ACENTOS for char in word)),  # acentos
        float(n_tokens),                               # palabras raras se parten en más tokens
        float(word in INTERROGATIVAS),
        float(any(char.isdigit() for char in word)),
        float(not word.isalpha()),                     # tiene signos o dígitos: "mcdonald's", "2x1"
    ]


def _position_features(n_words: int) -> torch.Tensor:
    """(n_palabras, 3): es la primera, es la última, posición relativa entre 0 y 1"""
    position = torch.arange(n_words, dtype=torch.float32)
    return torch.stack([position == 0, position == n_words - 1, position / max(n_words - 1, 1)], dim=1).float()


def _context_features(own: torch.Tensor, context: int) -> torch.Tensor:
    """Los atributos de las `context` palabras anteriores y siguientes de cada palabra, con
    ceros fuera de la instancia. (n_palabras, n_atributos) -> (n_palabras, 2 * context * n_atributos)"""
    n_words = own.size(0)
    padded = F.pad(own, (0, 0, context, context))  # filas de ceros arriba y abajo
    offsets = [offset for offset in range(-context, context + 1) if offset != 0]
    return torch.cat([padded[context + offset: context + offset + n_words] for offset in offsets], dim=1)


class ClassicFeatureExtractor(TextFeatureExtractor):
    """Feature Extractor sin embeddings"""

    def extract_words(self, tokens: list[str]) -> torch.Tensor:
        """Devuelve un tensor (n_palabras, n_features): atributos propios, posición y contexto"""
        n_tokens = np.bincount(token_word_ids(tokens))
        own = torch.tensor(
            [_word_features(word, n) for word, n in zip(split_words(tokens), n_tokens)], dtype=torch.float32
        )
        return torch.cat([own, _position_features(own.size(0)), _context_features(own, CONTEXT_WORDS)], dim=1)

    def extract(self, tokens: list[str]) -> torch.Tensor:
        """Devuelve un tensor (n_tokens, n_features): cada token con los atributos de su palabra"""
        return self.extract_words(tokens)[torch.from_numpy(token_word_ids(tokens))]
