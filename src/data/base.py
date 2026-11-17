from abc import abstractmethod

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BertTokenizer

from src.constants import EMBEDDINGS_MODEL, PUNCTUATION_MARKS

IGNORE_INDEX = -100


def strip_punctuation(word: str) -> str:
    """Quita los signos de puntuación de los bordes de la palabra.

    Parameters
    ----------
    word : str
        Palabra con su puntuación original.

    Returns
    -------
    str
        La palabra sin los signos de ``PUNCTUATION_MARKS`` al principio ni al final.

    Examples
    --------
    >>> strip_punctuation("¿Cuándo?")
    'Cuándo'
    """
    return word.strip(PUNCTUATION_MARKS)


def normalize_word(word: str) -> str:
    """Normaliza la palabra como la ven los modelos: sin puntuación en los bordes y en minúsculas.

    Parameters
    ----------
    word : str
        Palabra con su puntuación y capitalización originales.

    Returns
    -------
    str
        La palabra normalizada.

    Examples
    --------
    >>> normalize_word("¿Cuándo?")
    'cuándo'
    """
    return strip_punctuation(word).lower()


def collate_batch(
    batch: list[tuple[list[str], torch.Tensor]],
) -> tuple[list[list[str]], torch.Tensor]:
    """Agrupa instancias de largo variable en un batch.

    Parameters
    ----------
    batch : list[tuple[list[str], torch.Tensor]]
        Pares ``(tokens, etiquetas)`` tal como los devuelve ``TextDataset.__getitem__``.

    Returns
    -------
    tokens : list[list[str]]
        Los tokens de cada instancia.
    labels : torch.Tensor of shape (batch, max_tokens, ...)
        Las etiquetas, rellenadas con ``IGNORE_INDEX`` hasta el largo máximo.
    """
    tokens, labels = zip(*batch)
    return list(tokens), pad_sequence(list(labels), batch_first=True, padding_value=IGNORE_INDEX)


class TextDataset(Dataset):
    """Dataset de instancias de texto que tokeniza cada palabra normalizada.

    Las subclases definen qué etiqueta lleva cada token con ``_expected_output``.

    Parameters
    ----------
    samples : list[list[str]]
        Cada instancia es una lista de palabras con su puntuación y capitalización
        originales (ver ``src/data/samples.py``).

    Attributes
    ----------
    samples : list[list[str]]
        Las instancias recibidas.
    len : int
        Cantidad de instancias.
    tokenizer : BertTokenizer
        Tokenizador de ``EMBEDDINGS_MODEL``.
    """

    def __init__(self, samples: list[list[str]]) -> None:
        self.samples = samples
        self.len = len(self.samples)
        self.tokenizer = BertTokenizer.from_pretrained(EMBEDDINGS_MODEL)

    @abstractmethod
    def _expected_output(self, words: list[str], word_tokens: list[list[str]]) -> torch.Tensor:
        """Calcula la etiqueta de cada token.

        Parameters
        ----------
        words : list[str]
            Palabras originales de la instancia.
        word_tokens : list[list[str]]
            Tokens de cada palabra normalizada.

        Returns
        -------
        torch.Tensor of shape (n_tokens, ...)
            Una etiqueta (o una fila de etiquetas) por token.
        """
        raise NotImplementedError()

    def _tokenize_words(self, words: list[str]) -> list[list[str]]:
        """Tokeniza cada palabra normalizada (sin puntuación y en minúsculas).

        Parameters
        ----------
        words : list[str]
            Palabras originales de la instancia.

        Returns
        -------
        list[list[str]]
            Los tokens de cada palabra. Un signo suelto queda sin tokens.
        """
        return [self.tokenizer.tokenize(normalize_word(word)) for word in words]

    def __len__(self) -> int:
        """Devuelve la cantidad de instancias.

        Returns
        -------
        int
            Cantidad de instancias del dataset.
        """
        return self.len

    def __getitem__(self, idx: int) -> tuple[list[str], torch.Tensor]:
        """Tokeniza la instancia ``idx`` y calcula sus etiquetas.

        Parameters
        ----------
        idx : int
            Índice de la instancia.

        Returns
        -------
        tokens : list[str]
            Los tokens de todas las palabras de la instancia, en orden.
        y : torch.Tensor of shape (n_tokens, ...)
            La etiqueta de cada token (ver ``_expected_output``).
        """
        words = self.samples[idx]
        word_tokens = self._tokenize_words(words)
        tokens = [token for tokens in word_tokens for token in tokens]
        y = self._expected_output(words, word_tokens)
        return tokens, y
