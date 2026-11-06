import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from src.data import IGNORE_INDEX
from src.metrics.base import Metric


class CapitalizationMetric(Metric):

    def score(self, pred: np.ndarray, real: np.ndarray) -> float:
        """F1 macro a nivel token"""
        return float(f1_score(real, pred, average="macro", zero_division=0))

    def loss(self, logits: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """CrossEntropy ignorando el padding. Los logits tienen forma (..., n_clases)
        y las etiquetas (...)"""
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            real.reshape(-1),
            ignore_index=IGNORE_INDEX
        )
