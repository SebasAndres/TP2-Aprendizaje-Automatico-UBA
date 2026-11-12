import numpy as np
import torch
import torch.nn.functional as F

from src.data import IGNORE_INDEX
from src.metrics.base import Metric, WordLevelTask
from src.words import first_token_mask, majority_by_word

N_CAPITALIZATION_CLASSES = 4


class CapitalizationMetric(Metric):

    def word_level(self, pred: np.ndarray, real: np.ndarray, word_ids: np.ndarray) -> dict[str, WordLevelTask]:
        """Todos los tokens de una palabra comparten la etiqueta real,
        y la predicción de la palabra es la más votada entre sus tokens"""
        word_real = real[first_token_mask(word_ids)]
        word_pred = majority_by_word(pred, word_ids)
        return {"capitalizacion": (word_real, word_pred, N_CAPITALIZATION_CLASSES)}

    def loss(self, logits: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """CrossEntropy ignorando el padding. Los logits tienen forma (..., n_clases)
        y las etiquetas (...)"""
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            real.reshape(-1),
            ignore_index=IGNORE_INDEX
        )
