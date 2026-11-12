import numpy as np
import torch

from src.data.capitalization_dataset import N_CAPITALIZATION_CLASSES
from src.metrics.base import Metric, WordLevelTask, token_cross_entropy
from src.words import first_token_mask, majority_by_word


class CapitalizationMetric(Metric):
    tasks = ("capitalizacion",)

    def word_level(self, pred: np.ndarray, real: np.ndarray, word_ids: np.ndarray) -> dict[str, WordLevelTask]:
        """Todos los tokens de una palabra comparten la etiqueta real,
        y la predicción de la palabra es la más votada entre sus tokens"""
        word_real = real[first_token_mask(word_ids)]
        word_pred = majority_by_word(pred, word_ids)
        return {"capitalizacion": (word_real, word_pred, N_CAPITALIZATION_CLASSES)}

    def loss(self, logits: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """CrossEntropy ignorando el padding. logits: (batch, n_tokens, n_clases), real: (batch, n_tokens)"""
        return token_cross_entropy(logits, real)
