import numpy as np
import torch

from src.data.capitalization_dataset import N_CAPITALIZATION_CLASSES
from src.metrics.base import ClassWeights, Metric, WordLevelTask, token_cross_entropy
from src.words import first_token_mask, majority_by_word


class CapitalizationMetric(Metric):
    tasks = ("capitalizacion",)

    def word_labels(self, labels: np.ndarray, word_ids: np.ndarray) -> np.ndarray:
        """Todos los tokens de una palabra comparten la etiqueta real: la del primero"""
        return labels[first_token_mask(word_ids)]

    def word_level(self, pred: np.ndarray, real: np.ndarray, word_ids: np.ndarray) -> dict[str, WordLevelTask]:
        """La predicción de la palabra es la más votada entre sus tokens"""
        word_real = self.word_labels(real, word_ids)
        word_pred = majority_by_word(pred, word_ids)
        return {"capitalizacion": (word_real, word_pred, N_CAPITALIZATION_CLASSES)}

    def loss(self, logits: torch.Tensor, real: torch.Tensor, class_weights: ClassWeights | None = None) -> torch.Tensor:
        """CrossEntropy ignorando el padding. logits: (batch, n_tokens, n_clases), real: (batch, n_tokens)
        class_weights: (1, n_clases) o None"""
        return token_cross_entropy(logits, real, None if class_weights is None else class_weights[0])
