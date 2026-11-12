import numpy as np
import torch

from src.data.punctuation_dataset import FINAL_LABELS, INITIAL_LABELS
from src.metrics.base import Metric, WordLevelTask, multitask_cross_entropy
from src.words import first_token_mask, last_token_mask


class PunctuationMetric(Metric):
    tasks = ("punt_inicial", "punt_final")

    def word_level(self, pred: np.ndarray, real: np.ndarray, word_ids: np.ndarray) -> dict[str, WordLevelTask]:
        """La puntuación inicial se lee del primer token de la palabra y la final del último,
        que es donde las asigna el dataset"""
        first = first_token_mask(word_ids)
        last = last_token_mask(word_ids)
        return {
            "punt_inicial": (real[first, 0], pred[first, 0], len(INITIAL_LABELS)),
            "punt_final": (real[last, 1], pred[last, 1], len(FINAL_LABELS))
        }

    def loss(self, logits: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """Suma de la CrossEntropy de la puntuación inicial y la final, ignorando el padding.
        logits: (batch, n_tokens, 2 tareas, 4 clases), real: (batch, n_tokens, 2 tareas)"""
        return multitask_cross_entropy(logits, real)
