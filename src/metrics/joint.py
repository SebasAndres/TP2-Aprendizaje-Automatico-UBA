import numpy as np
import torch

from src.metrics.base import Metric, WordLevelTask, multitask_cross_entropy
from src.metrics.capitalization import CapitalizationMetric
from src.metrics.punctuation import PunctuationMetric


class JointMetric(Metric):
    """Capitalización y puntuación a la vez. Las etiquetas por token son
    <capitalización, puntuación inicial, puntuación final>"""

    capitalization = CapitalizationMetric()
    punctuation = PunctuationMetric()
    tasks = capitalization.tasks + punctuation.tasks

    def word_level(self, pred: np.ndarray, real: np.ndarray, word_ids: np.ndarray) -> dict[str, WordLevelTask]:
        """La columna 0 se evalúa como capitalización y las columnas 1 y 2 como puntuación"""
        return {
            **self.capitalization.word_level(pred[:, 0], real[:, 0], word_ids),
            **self.punctuation.word_level(pred[:, 1:], real[:, 1:], word_ids),
        }

    def loss(self, logits: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """Suma de la CrossEntropy de cada tarea, ignorando el padding.
        logits: (batch, n_tokens, 3 tareas, n_clases), real: (batch, n_tokens, 3 tareas)"""
        return multitask_cross_entropy(logits, real)
