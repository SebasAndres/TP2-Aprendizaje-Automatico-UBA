"""Dataset de las redes conjuntas: las 3 etiquetas de cada token a la vez."""

import torch

from src.data.capitalization_dataset import CapitalizationDataset
from src.data.punctuation_dataset import PunctuationDataset


class JointDataset(CapitalizationDataset, PunctuationDataset):
    """Cada token tiene 3 etiquetas: <capitalización, puntuación inicial, puntuación final>."""

    def _expected_output(self, words: list[str], word_tokens: list[list[str]]) -> torch.Tensor:
        """Junta las etiquetas de capitalización y de puntuación de cada token.

        Parameters
        ----------
        words : list[str]
            Palabras originales de la instancia.
        word_tokens : list[list[str]]
            Tokens de cada palabra normalizada.

        Returns
        -------
        torch.Tensor of shape (n_tokens, 3)
            Cada fila es ``[capitalización, inicial, final]``.
        """
        capitalization = CapitalizationDataset._expected_output(self, words, word_tokens)  # (n_tokens,)
        punctuation = PunctuationDataset._expected_output(self, words, word_tokens)  # (n_tokens, 2)
        return torch.cat([capitalization.unsqueeze(1), punctuation], dim=1)  # (n_tokens, 3)
