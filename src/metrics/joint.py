import numpy as np
import torch
import torch.nn.functional as F

from src.data import IGNORE_INDEX
from src.metrics.base import Metric, WordLevelTask
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
        return sum(
            F.cross_entropy(
                logits[:, :, task].reshape(-1, logits.size(-1)),
                real[:, :, task].reshape(-1),
                ignore_index=IGNORE_INDEX
            )
            for task in range(logits.size(2))
        )
