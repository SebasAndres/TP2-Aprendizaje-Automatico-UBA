import numpy as np
import torch

from src.metrics.base import Metric


class PunctuationMetric(Metric):

    def score(self, pred: np.ndarray, real: np.ndarray) -> float:
        raise NotImplementedError("PunctuationMetric.score todavía no está implementado")

    def loss(self, logits: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("PunctuationMetric.loss todavía no está implementado")
