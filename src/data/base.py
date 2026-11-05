import gzip
from abc import abstractmethod
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BertTokenizer

from src.constants import EMBEDDINGS_MODEL, PUNCTUATION_MARKS
from src.features import TextFeatureExtractor

IGNORE_INDEX = -100


def strip_punctuation(word: str) -> str:
    """Quita los signos de puntuación de los bordes de la palabra"""
    return word.strip(PUNCTUATION_MARKS)


def collate_batch(
    batch: list[tuple[list[str], torch.Tensor]],
) -> tuple[list[list[str]], torch.Tensor]:
    """Agrupa oraciones de largo variable, rellenando las etiquetas con IGNORE_INDEX"""
    tokens, labels = zip(*batch)
    return list(tokens), pad_sequence(list(labels), batch_first=True, padding_value=IGNORE_INDEX)


class TextDataset(Dataset):

    def __init__(
        self, 
        path: str,
    ):
        self.path = path
        with gzip.open(self.path, mode="rt", encoding="utf-8") as f:
            self.data = f.read().splitlines()
        self.len = len(self.data)
        self.tokenizer = BertTokenizer.from_pretrained(EMBEDDINGS_MODEL)

    @abstractmethod
    def _expected_output(self, words: list[str], word_tokens: list[list[str]]):
        """
        Recibe las palabras originales del texto y los tokens de cada palabra normalizada,
        y devuelve una etiqueta por token
        """
        raise NotImplementedError()

    def _extract_features(self, tokens: list[str]):
        return self.feature_extractor.extract(tokens)

    def _tokenize_words(self, words: list[str]) -> list[list[str]]:
        """Tokeniza cada palabra normalizada (sin puntuación y en minúsculas)"""
        return [self.tokenizer.tokenize(strip_punctuation(word).lower()) for word in words]

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int):
        words = self.data[idx].split()
        word_tokens = self._tokenize_words(words)
        tokens = [token for tokens in word_tokens for token in tokens]
        y = self._expected_output(words, word_tokens)
        return tokens, y
