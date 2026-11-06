from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.metrics import f1_score


def macro_f1(real: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """F1 macro sobre todas las clases, aunque alguna no aparezca en estos datos"""
    return float(f1_score(real, pred, average="macro", labels=range(n_classes), zero_division=0))


class Metric(ABC):

    @abstractmethod
    def score(self, pred: np.ndarray, real: np.ndarray, word_ids: np.ndarray) -> dict[str, float]:
        """Compara las clases predichas con las reales a nivel palabra, sin padding, y devuelve
        un F1 macro por tarea. word_ids indica la palabra de cada token"""
        raise NotImplementedError

    @abstractmethod
    def loss(self, logits: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """Función de pérdida a optimizar durante el entrenamiento"""
        raise NotImplementedError
