import numpy as np
import torch

from src.data.punctuation_dataset import FINAL_LABELS, INITIAL_LABELS
from src.metrics.base import ClassWeights, Metric, WordLevelTask, multitask_cross_entropy
from src.words import first_token_mask, last_token_mask


class PunctuationMetric(Metric):
    tasks = ("punt_inicial", "punt_final")

    def word_labels(self, labels: np.ndarray, word_ids: np.ndarray) -> np.ndarray:
        """[inicial, final] de cada palabra: la inicial del primer token y la final del último,
        que es donde las asigna el dataset. -> (n_palabras, 2)"""
        return np.stack([labels[first_token_mask(word_ids), 0], labels[last_token_mask(word_ids), 1]], axis=1)

    def word_level(self, pred: np.ndarray, real: np.ndarray, word_ids: np.ndarray) -> dict[str, WordLevelTask]:
        """Las predicciones se leen de los mismos tokens que las etiquetas reales"""
        word_real, word_pred = self.word_labels(real, word_ids), self.word_labels(pred, word_ids)
        return {
            "punt_inicial": (word_real[:, 0], word_pred[:, 0], len(INITIAL_LABELS)),
            "punt_final": (word_real[:, 1], word_pred[:, 1], len(FINAL_LABELS))
        }

    def loss(
        self, logits: torch.Tensor, real: torch.Tensor, class_weights: ClassWeights | None = None
    ) -> torch.Tensor:
        """Suma de la CrossEntropy de la puntuación inicial y la final, ignorando el padding.
        logits: (batch, n_tokens, 2 tareas, 4 clases), real: (batch, n_tokens, 2 tareas)"""
        return multitask_cross_entropy(logits, real, class_weights)
