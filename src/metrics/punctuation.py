import numpy as np
import torch

from src.data.punctuation_dataset import FINAL_LABELS, INITIAL_LABELS
from src.metrics.base import Metric, macro_f1
from src.metrics.words import first_token_mask, last_token_mask


class PunctuationMetric(Metric):

    def score(self, pred: np.ndarray, real: np.ndarray, word_ids: np.ndarray) -> dict[str, float]:
        """F1 macro a nivel palabra de cada tarea. La puntuación inicial se lee del primer token
        de la palabra y la final del último, que es donde las asigna el dataset"""
        first = first_token_mask(word_ids)
        last = last_token_mask(word_ids)
        return {
            "punt_inicial": macro_f1(real[first, 0], pred[first, 0], len(INITIAL_LABELS)),
            "punt_final": macro_f1(real[last, 1], pred[last, 1], len(FINAL_LABELS))
        }

    def loss(self, logits: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("PunctuationMetric.loss todavía no está implementado")
