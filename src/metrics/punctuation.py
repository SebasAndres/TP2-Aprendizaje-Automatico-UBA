import numpy as np
import torch
import torch.nn.functional as F

from src.data.punctuation_dataset import FINAL_LABELS, INITIAL_LABELS
from src.metrics.base import Metric, WordLevelTask
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
        # logits: (batch, n_tokens, 2_tareas, 4_clases)
        # real: (batch, n_tokens, 2_tareas)
        
        # Aplanamos los tensores para PyTorch: (N, C) y (N,)
        logits_ini = logits[:, :, 0, :].reshape(-1, 4)
        real_ini = real[:, :, 0].reshape(-1)
        
        logits_fin = logits[:, :, 1, :].reshape(-1, 4)
        real_fin = real[:, :, 1].reshape(-1)
        
        # Calculamos la entropía cruzada ignorando el padding (-100)
        loss_ini = F.cross_entropy(logits_ini, real_ini, ignore_index=-100)
        loss_fin = F.cross_entropy(logits_fin, real_fin, ignore_index=-100)
        
        # El error total de la red es la suma de ambos errores
        return loss_ini + loss_fin
