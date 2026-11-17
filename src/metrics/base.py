from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src.data import IGNORE_INDEX
from src.data.capitalization_dataset import CAPITALIZATION_NAMES
from src.data.punctuation_dataset import FINAL_LABELS, INITIAL_LABELS

WordLevelTask = tuple[np.ndarray, np.ndarray, int]
"""(etiquetas reales, predicciones, cantidad de clases) de una tarea, a nivel palabra"""

PerClass = dict[str, dict[int, tuple[float, float]]]
"""Por tarea, por clase: (precision, recall)"""

CLASS_NAMES = {
    "capitalizacion": CAPITALIZATION_NAMES,
    "punt_inicial": [label or "nada" for label in INITIAL_LABELS],
    "punt_final": [label or "nada" for label in FINAL_LABELS],
}


def class_name(task: str, c: int) -> str:
    """Devuelve el nombre legible de una clase, para gráficos y salidas.

    Parameters
    ----------
    task : str
        Nombre de la tarea (``"capitalizacion"``, ``"punt_inicial"`` o ``"punt_final"``).
    c : int
        Índice de la clase.

    Returns
    -------
    str
        El nombre de la clase, o ``"clase c"`` si la tarea o la clase no tienen nombre.
    """
    names = CLASS_NAMES.get(task, [])
    return names[c] if c < len(names) else f"clase {c}"


def macro_f1(real: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """Calcula el F1 macro sobre todas las clases, aunque alguna no aparezca en estos datos.

    Parameters
    ----------
    real : np.ndarray
        Etiquetas reales.
    pred : np.ndarray
        Clases predichas.
    n_classes : int
        Cantidad de clases de la tarea.

    Returns
    -------
    float
        Promedio del F1 de las clases ``0..n_classes-1`` (0 para las que no tienen datos).
    """
    return float(f1_score(real, pred, average="macro", labels=range(n_classes), zero_division=0))


def precision_recall(real: np.ndarray, pred: np.ndarray, n_classes: int) -> dict[int, tuple[float, float]]:
    """Calcula la precision y el recall de cada clase, aunque alguna no aparezca en estos datos.

    Parameters
    ----------
    real : np.ndarray
        Etiquetas reales.
    pred : np.ndarray
        Clases predichas.
    n_classes : int
        Cantidad de clases de la tarea.

    Returns
    -------
    dict[int, tuple[float, float]]
        ``(precision, recall)`` de cada clase.
    """
    precision, recall, _, _ = precision_recall_fscore_support(real, pred, labels=range(n_classes), zero_division=0)
    return {c: (float(precision[c]), float(recall[c])) for c in range(n_classes)}


ClassWeights = torch.Tensor
"""Peso de cada clase en la loss, por columna de etiquetas: (n_columnas, n_clases).
Ver src/training/weights.py"""


def token_cross_entropy(
    logits: torch.Tensor, real: torch.Tensor, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """Calcula la Focal Loss por token, ignorando el padding.

    Es la CrossEntropy de cada token multiplicada por ``(1 - p_t) ** 2``, con ``p_t``
    la probabilidad de la clase real: los tokens que la red ya acierta con confianza
    pesan menos.

    Parameters
    ----------
    logits : torch.Tensor of shape (..., n_clases)
        Salida de la red para una tarea.
    real : torch.Tensor of shape (...)
        Etiquetas reales, con ``IGNORE_INDEX`` en el padding.
    weight : torch.Tensor of shape (n_clases,) or None, default=None
        Peso de cada clase. Se recorta o se completa con 1 hasta ``n_clases``.

    Returns
    -------
    torch.Tensor
        Escalar con la loss promedio sobre los tokens reales (0 si no hay ninguno).
    """
    n_classes = logits.size(-1)
    if weight is not None:
        weight = F.pad(weight[:n_classes], (0, max(0, n_classes - weight.numel())), value=1.0).to(logits.device)
    
    # 1. Calculamos la CrossEntropy sin reducir para retener el error de cada token
    ce_loss = F.cross_entropy(
        logits.reshape(-1, n_classes), real.reshape(-1), 
        weight=weight, ignore_index=IGNORE_INDEX, reduction='none'
    )
    
    # 2. Extraemos la probabilidad (p_t) y aplicamos el factor de Focal Loss
    pt = torch.exp(-ce_loss)
    gamma = 2.0
    focal_loss = ((1 - pt) ** gamma) * ce_loss
    
    # 3. Promediamos solo sobre los tokens reales (excluyendo el padding de ceros)
    valid_mask = (real.reshape(-1) != IGNORE_INDEX)
    if valid_mask.any():
        return focal_loss[valid_mask].mean()
    return torch.tensor(0.0, device=logits.device)


def multitask_cross_entropy(
    logits: torch.Tensor, real: torch.Tensor, class_weights: ClassWeights | None = None
) -> torch.Tensor:
    """Suma la loss de cada tarea, ignorando el padding.

    Parameters
    ----------
    logits : torch.Tensor of shape (batch, n_tokens, n_tareas, n_clases)
        Salida de la red.
    real : torch.Tensor of shape (batch, n_tokens, n_tareas)
        Etiquetas reales, con ``IGNORE_INDEX`` en el padding.
    class_weights : ClassWeights or None, default=None
        Peso de cada clase de cada tarea, ``(n_tareas, n_clases)``, o ``None``.

    Returns
    -------
    torch.Tensor
        Escalar con la suma de ``token_cross_entropy`` de cada tarea.
    """
    return sum(
        token_cross_entropy(
            logits[:, :, task], real[:, :, task], None if class_weights is None else class_weights[task]
        )
        for task in range(logits.size(2))
    )


def scores_from_tasks(tasks: dict[str, WordLevelTask]) -> dict[str, float]:
    """Calcula el F1 macro de cada tarea.

    Parameters
    ----------
    tasks : dict[str, WordLevelTask]
        Etiquetas reales, predicciones y cantidad de clases de cada tarea.

    Returns
    -------
    dict[str, float]
        F1 macro de cada tarea.
    """
    return {task: macro_f1(r, p, n) for task, (r, p, n) in tasks.items()}


def per_class_from_tasks(tasks: dict[str, WordLevelTask]) -> PerClass:
    """Calcula la precision y el recall de cada clase de cada tarea.

    Parameters
    ----------
    tasks : dict[str, WordLevelTask]
        Etiquetas reales, predicciones y cantidad de clases de cada tarea.

    Returns
    -------
    PerClass
        Por tarea, ``(precision, recall)`` de cada clase.
    """
    return {task: precision_recall(r, p, n) for task, (r, p, n) in tasks.items()}


class Metric(ABC):
    """Métrica y loss de un problema, a nivel palabra.

    Attributes
    ----------
    tasks : tuple[str, ...]
        Nombres de las tareas que evalúa, en el orden en que las devuelve ``word_level``.
    """

    tasks: tuple[str, ...]

    @abstractmethod
    def word_labels(self, labels: np.ndarray, word_ids: np.ndarray) -> np.ndarray:
        """Pasa las etiquetas reales por token a una por palabra.

        Las toma del token donde las asigna el dataset. Los modelos clásicos se entrenan
        con estas etiquetas.

        Parameters
        ----------
        labels : np.ndarray of shape (n_tokens,) or (n_tokens, n_columnas)
            Etiquetas reales de cada token.
        word_ids : np.ndarray of shape (n_tokens,)
            Índice de palabra de cada token.

        Returns
        -------
        np.ndarray of shape (n_palabras,) or (n_palabras, n_columnas)
            La etiqueta de cada palabra.
        """
        raise NotImplementedError

    @abstractmethod
    def word_level(self, pred: np.ndarray, real: np.ndarray, word_ids: np.ndarray) -> dict[str, WordLevelTask]:
        """Pasa las clases por token (sin padding) a nivel palabra, separadas por tarea.

        Parameters
        ----------
        pred : np.ndarray of shape (n_tokens,) or (n_tokens, n_columnas)
            Clases predichas de cada token, sin padding.
        real : np.ndarray of shape (n_tokens,) or (n_tokens, n_columnas)
            Etiquetas reales de cada token, sin padding.
        word_ids : np.ndarray of shape (n_tokens,)
            Índice de palabra de cada token.

        Returns
        -------
        dict[str, WordLevelTask]
            Etiquetas reales, predicciones y cantidad de clases de cada tarea, por palabra.
        """
        raise NotImplementedError

    def score(self, pred: np.ndarray, real: np.ndarray, word_ids: np.ndarray) -> dict[str, float]:
        """Calcula el F1 macro a nivel palabra de cada tarea.

        Parameters
        ----------
        pred : np.ndarray of shape (n_tokens,) or (n_tokens, n_columnas)
            Clases predichas de cada token, sin padding.
        real : np.ndarray of shape (n_tokens,) or (n_tokens, n_columnas)
            Etiquetas reales de cada token, sin padding.
        word_ids : np.ndarray of shape (n_tokens,)
            Índice de palabra de cada token.

        Returns
        -------
        dict[str, float]
            F1 macro de cada tarea.
        """
        return scores_from_tasks(self.word_level(pred, real, word_ids))

    def per_class(self, pred: np.ndarray, real: np.ndarray, word_ids: np.ndarray) -> PerClass:
        """Calcula la precision y el recall a nivel palabra de cada clase de cada tarea.

        Parameters
        ----------
        pred : np.ndarray of shape (n_tokens,) or (n_tokens, n_columnas)
            Clases predichas de cada token, sin padding.
        real : np.ndarray of shape (n_tokens,) or (n_tokens, n_columnas)
            Etiquetas reales de cada token, sin padding.
        word_ids : np.ndarray of shape (n_tokens,)
            Índice de palabra de cada token.

        Returns
        -------
        PerClass
            Por tarea, ``(precision, recall)`` de cada clase.
        """
        return per_class_from_tasks(self.word_level(pred, real, word_ids))

    @abstractmethod
    def loss(self, logits: torch.Tensor, real: torch.Tensor, class_weights: ClassWeights | None = None) -> torch.Tensor:
        """Calcula la función de pérdida a optimizar durante el entrenamiento.

        Parameters
        ----------
        logits : torch.Tensor of shape (batch, n_tokens, ..., n_clases)
            Salida de la red.
        real : torch.Tensor of shape (batch, n_tokens, ...)
            Etiquetas reales, con ``IGNORE_INDEX`` en el padding.
        class_weights : ClassWeights or None, default=None
            Peso de cada clase por columna de ``real``, ``(n_columnas, n_clases)``, o
            ``None`` para no pesar.

        Returns
        -------
        torch.Tensor
            Escalar con la loss del batch.
        """
        raise NotImplementedError
