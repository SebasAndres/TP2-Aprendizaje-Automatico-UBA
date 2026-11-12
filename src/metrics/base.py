from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

WordLevelTask = tuple[np.ndarray, np.ndarray, int]
"""(etiquetas reales, predicciones, cantidad de clases) de una tarea, a nivel palabra"""

PerClass = dict[str, dict[int, tuple[float, float]]]
"""Por tarea, por clase: (precision, recall)"""


def macro_f1(real: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """F1 macro sobre todas las clases, aunque alguna no aparezca en estos datos"""
    return float(f1_score(real, pred, average="macro", labels=range(n_classes), zero_division=0))


def precision_recall(real: np.ndarray, pred: np.ndarray, n_classes: int) -> dict[int, tuple[float, float]]:
    """(precision, recall) de cada clase, aunque alguna no aparezca en estos datos"""
    precision, recall, _, _ = precision_recall_fscore_support(
        real, pred, labels=range(n_classes), zero_division=0
    )
    return {c: (float(precision[c]), float(recall[c])) for c in range(n_classes)}


class Metric(ABC):

    @abstractmethod
    def word_level(self, pred: np.ndarray, real: np.ndarray, word_ids: np.ndarray) -> dict[str, WordLevelTask]:
        """Pasa las clases por token (sin padding) a nivel palabra, separadas por tarea.
        word_ids indica la palabra de cada token"""
        raise NotImplementedError

    def score(self, pred: np.ndarray, real: np.ndarray, word_ids: np.ndarray) -> dict[str, float]:
        """F1 macro a nivel palabra de cada tarea"""
        tasks = self.word_level(pred, real, word_ids)
        return {task: macro_f1(r, p, n) for task, (r, p, n) in tasks.items()}

    def per_class(self, pred: np.ndarray, real: np.ndarray, word_ids: np.ndarray) -> PerClass:
        """(precision, recall) a nivel palabra de cada clase de cada tarea"""
        tasks = self.word_level(pred, real, word_ids)
        return {task: precision_recall(r, p, n) for task, (r, p, n) in tasks.items()}

    @abstractmethod
    def loss(self, logits: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """Función de pérdida a optimizar durante el entrenamiento"""
        raise NotImplementedError
