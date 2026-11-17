import numpy as np
import torch

from src.data.capitalization_dataset import N_CAPITALIZATION_CLASSES
from src.metrics.base import ClassWeights, Metric, WordLevelTask, token_cross_entropy
from src.words import first_token_mask, majority_by_word


class CapitalizationMetric(Metric):
    """Métrica del problema de capitalización: una tarea, ``"capitalizacion"``."""

    tasks = ("capitalizacion",)

    def word_labels(self, labels: np.ndarray, word_ids: np.ndarray) -> np.ndarray:
        """Toma la etiqueta real de cada palabra de su primer token.

        Todos los tokens de una palabra comparten la etiqueta real.

        Parameters
        ----------
        labels : np.ndarray of shape (n_tokens,)
            Capitalización real de cada token.
        word_ids : np.ndarray of shape (n_tokens,)
            Índice de palabra de cada token.

        Returns
        -------
        np.ndarray of shape (n_palabras,)
            Capitalización real de cada palabra.
        """
        return labels[first_token_mask(word_ids)]

    def word_level(self, pred: np.ndarray, real: np.ndarray, word_ids: np.ndarray) -> dict[str, WordLevelTask]:
        """Pasa la capitalización a nivel palabra: la predicción es la más votada entre sus tokens.

        Parameters
        ----------
        pred : np.ndarray of shape (n_tokens,)
            Clase predicha de cada token.
        real : np.ndarray of shape (n_tokens,)
            Clase real de cada token.
        word_ids : np.ndarray of shape (n_tokens,)
            Índice de palabra de cada token.

        Returns
        -------
        dict[str, WordLevelTask]
            ``{"capitalizacion": (reales, predichas, N_CAPITALIZATION_CLASSES)}`` por palabra.
        """
        word_real = self.word_labels(real, word_ids)
        word_pred = majority_by_word(pred, word_ids)
        return {"capitalizacion": (word_real, word_pred, N_CAPITALIZATION_CLASSES)}

    def loss(self, logits: torch.Tensor, real: torch.Tensor, class_weights: ClassWeights | None = None) -> torch.Tensor:
        """Calcula la loss de capitalización, ignorando el padding.

        Parameters
        ----------
        logits : torch.Tensor of shape (batch, n_tokens, n_clases)
            Salida de la red.
        real : torch.Tensor of shape (batch, n_tokens)
            Clase real de cada token, con ``IGNORE_INDEX`` en el padding.
        class_weights : ClassWeights or None, default=None
            Peso de cada clase, ``(1, n_clases)``, o ``None`` para no pesar.

        Returns
        -------
        torch.Tensor
            Escalar con la loss del batch (ver ``token_cross_entropy``).
        """
        return token_cross_entropy(logits, real, None if class_weights is None else class_weights[0])
