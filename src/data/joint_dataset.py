"""Dataset de las redes conjuntas: las 3 etiquetas de cada token a la vez."""

import torch

from src.data.capitalization_dataset import CapitalizationDataset
from src.data.punctuation_dataset import PunctuationDataset


class JointDataset(CapitalizationDataset, PunctuationDataset):
    """Cada token tiene 3 etiquetas: <capitalización, puntuación inicial, puntuación final>"""

    def _expected_output(self, words: list[str], word_tokens: list[list[str]]) -> torch.Tensor:
        capitalization = CapitalizationDataset._expected_output(self, words, word_tokens)  # (n_tokens,)
        punctuation = PunctuationDataset._expected_output(self, words, word_tokens)  # (n_tokens, 2)
        return torch.cat([capitalization.unsqueeze(1), punctuation], dim=1)  # (n_tokens, 3)
