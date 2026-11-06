from abc import ABC, abstractmethod

import numpy as np
import torch


class Metric(ABC):

    @abstractmethod
    def score(self, pred: np.ndarray, real: np.ndarray) -> float:
        """Compara las clases predichas con las reales, sin padding. Se usa para comparar modelos"""
        raise NotImplementedError

    @abstractmethod
    def loss(self, logits: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """Función de pérdida a optimizar durante el entrenamiento"""
        raise NotImplementedError
