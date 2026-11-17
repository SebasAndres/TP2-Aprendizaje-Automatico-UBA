import numpy as np
import torch

from src.data.punctuation_dataset import FINAL_LABELS, INITIAL_LABELS
from src.metrics.base import ClassWeights, Metric, WordLevelTask, multitask_cross_entropy
from src.words import first_token_mask, last_token_mask


class PunctuationMetric(Metric):
    """Métrica del problema de puntuación: tareas ``"punt_inicial"`` y ``"punt_final"``."""

    tasks = ("punt_inicial", "punt_final")

    def word_labels(self, labels: np.ndarray, word_ids: np.ndarray) -> np.ndarray:
        """Toma la inicial del primer token de cada palabra y la final del último.

        Es donde las asigna el dataset.

        Parameters
        ----------
        labels : np.ndarray of shape (n_tokens, 2)
            ``[inicial, final]`` de cada token.
        word_ids : np.ndarray of shape (n_tokens,)
            Índice de palabra de cada token.

        Returns
        -------
        np.ndarray of shape (n_palabras, 2)
            ``[inicial, final]`` de cada palabra.
        """
        return np.stack([labels[first_token_mask(word_ids), 0], labels[last_token_mask(word_ids), 1]], axis=1)

    def word_level(self, pred: np.ndarray, real: np.ndarray, word_ids: np.ndarray) -> dict[str, WordLevelTask]:
        """Pasa la puntuación a nivel palabra, leyendo las predicciones de los mismos tokens que las etiquetas.

        Parameters
        ----------
        pred : np.ndarray of shape (n_tokens, 2)
            ``[inicial, final]`` predichas de cada token.
        real : np.ndarray of shape (n_tokens, 2)
            ``[inicial, final]`` reales de cada token.
        word_ids : np.ndarray of shape (n_tokens,)
            Índice de palabra de cada token.

        Returns
        -------
        dict[str, WordLevelTask]
            ``"punt_inicial"`` y ``"punt_final"``: (reales, predichas, cantidad de clases).
        """
        word_real, word_pred = self.word_labels(real, word_ids), self.word_labels(pred, word_ids)
        return {
            "punt_inicial": (word_real[:, 0], word_pred[:, 0], len(INITIAL_LABELS)),
            "punt_final": (word_real[:, 1], word_pred[:, 1], len(FINAL_LABELS)),
        }

    def loss(self, logits: torch.Tensor, real: torch.Tensor, class_weights: ClassWeights | None = None) -> torch.Tensor:
        """Suma la loss de la puntuación inicial y la final, ignorando el padding.

        Parameters
        ----------
        logits : torch.Tensor of shape (batch, n_tokens, 2, 4)
            Salida de la red: 2 tareas, 4 clases.
        real : torch.Tensor of shape (batch, n_tokens, 2)
            ``[inicial, final]`` reales, con ``IGNORE_INDEX`` en el padding.
        class_weights : ClassWeights or None, default=None
            Peso de cada clase de cada tarea, ``(2, n_clases)``, o ``None``.

        Returns
        -------
        torch.Tensor
            Escalar con la loss del batch.
        """
        return multitask_cross_entropy(logits, real, class_weights)
