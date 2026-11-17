import numpy as np
import torch

from src.metrics.base import ClassWeights, Metric, WordLevelTask, multitask_cross_entropy
from src.metrics.capitalization import CapitalizationMetric
from src.metrics.punctuation import PunctuationMetric


class JointMetric(Metric):
    """Capitalización y puntuación a la vez.

    Las etiquetas por token son ``<capitalización, puntuación inicial, puntuación final>``.

    Attributes
    ----------
    capitalization : CapitalizationMetric
        Métrica de la columna 0.
    punctuation : PunctuationMetric
        Métrica de las columnas 1 y 2.
    tasks : tuple[str, ...]
        ``("capitalizacion", "punt_inicial", "punt_final")``.
    """

    capitalization = CapitalizationMetric()
    punctuation = PunctuationMetric()
    tasks = capitalization.tasks + punctuation.tasks

    def word_labels(self, labels: np.ndarray, word_ids: np.ndarray) -> np.ndarray:
        """Pasa las 3 etiquetas reales de cada token a una fila por palabra.

        Parameters
        ----------
        labels : np.ndarray of shape (n_tokens, 3)
            ``[capitalización, inicial, final]`` de cada token.
        word_ids : np.ndarray of shape (n_tokens,)
            Índice de palabra de cada token.

        Returns
        -------
        np.ndarray of shape (n_palabras, 3)
            ``[capitalización, inicial, final]`` de cada palabra.
        """
        capitalization = self.capitalization.word_labels(labels[:, 0], word_ids)
        punctuation = self.punctuation.word_labels(labels[:, 1:], word_ids)
        return np.column_stack([capitalization, punctuation])

    def word_level(self, pred: np.ndarray, real: np.ndarray, word_ids: np.ndarray) -> dict[str, WordLevelTask]:
        """Evalúa la columna 0 como capitalización y las columnas 1 y 2 como puntuación.

        Parameters
        ----------
        pred : np.ndarray of shape (n_tokens, 3)
            Clases predichas de cada token.
        real : np.ndarray of shape (n_tokens, 3)
            Clases reales de cada token.
        word_ids : np.ndarray of shape (n_tokens,)
            Índice de palabra de cada token.

        Returns
        -------
        dict[str, WordLevelTask]
            Las 3 tareas a nivel palabra.
        """
        return {
            **self.capitalization.word_level(pred[:, 0], real[:, 0], word_ids),
            **self.punctuation.word_level(pred[:, 1:], real[:, 1:], word_ids),
        }

    def loss(self, logits: torch.Tensor, real: torch.Tensor, class_weights: ClassWeights | None = None) -> torch.Tensor:
        """Suma la loss de cada tarea, ignorando el padding.

        Parameters
        ----------
        logits : torch.Tensor of shape (batch, n_tokens, 3, n_clases)
            Salida de la red conjunta.
        real : torch.Tensor of shape (batch, n_tokens, 3)
            Etiquetas reales, con ``IGNORE_INDEX`` en el padding.
        class_weights : ClassWeights or None, default=None
            Peso de cada clase de cada tarea, ``(3, n_clases)``, o ``None``.

        Returns
        -------
        torch.Tensor
            Escalar con la loss del batch.
        """
        return multitask_cross_entropy(logits, real, class_weights)
