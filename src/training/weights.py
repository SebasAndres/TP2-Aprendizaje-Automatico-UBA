"""Pesos por clase para la CrossEntropy de las redes.

La mayoría de los tokens no lleva puntuación ni mayúscula, así que una red que siempre
predice "nada"/"minúscula" ya tiene una loss baja. Con pesos, equivocarse en una clase
rara cuesta más que equivocarse en una común.

Los pesos se usan solo en la loss (la de train y la que se reporta sobre test). La F1
macro con la que se evalúan los modelos no los usa: ya promedia las clases por igual.

El peso de la clase c de una tarea es

    w_c = (N / (K · n_c)) ** power

con N los tokens etiquetados de la tarea, K las clases que aparecen en train y n_c los
tokens de la clase c.

- ``power = 1``: "balanced", cada clase pesa lo mismo en total (w_c · n_c = N / K). Con
  clases muy raras (¿ es ~1% de los tokens, MAYÚS ~0.1%) los pesos llegan a 50-250 y la
  red tiende a predecir la clase rara de más.
- ``power = 0.5``: raíz de lo anterior, un punto medio (pesos de ~0.5 a ~15).
- ``power = 0``: sin pesos.
"""

import logging

import torch
from torch.utils.data import Dataset

from src.data import IGNORE_INDEX

logger = logging.getLogger(__name__)


def label_counts(dataset: Dataset) -> torch.Tensor:
    """Cuenta los tokens de cada clase en cada columna de etiquetas.

    Parameters
    ----------
    dataset : Dataset
        Dataset de pares ``(x, y)``. ``y`` tiene forma ``(n_tokens,)`` para una sola tarea
        o ``(n_tokens, n_columnas)`` para varias.

    Returns
    -------
    torch.Tensor
        Tensor de enteros de forma ``(n_columnas, n_clases)``, con ``n_clases`` la mayor
        etiqueta del dataset + 1. Los tokens con ``IGNORE_INDEX`` (padding) no se
        cuentan. Una tarea con menos clases que otra queda con ceros al
        final de su fila.
    """
    labels = torch.cat([y.reshape(len(y), -1) for _, y in dataset])  # (n_tokens, n_columnas)
    n_classes = int(labels.max()) + 1
    return torch.stack([torch.bincount(column[column != IGNORE_INDEX], minlength=n_classes) for column in labels.T])


def class_weights(dataset: Dataset, power: float) -> torch.Tensor | None:
    """Calcula el peso de cada clase de cada tarea para la CrossEntropy.

    Por cada columna, ``w_c = (N / (K · n_c)) ** power`` (ver el docstring del módulo).

    Parameters
    ----------
    dataset : Dataset
        Dataset de train. Los pesos salen de sus frecuencias, nunca de las de test.
    power : float
        Exponente: 1 = "balanced", 0.5 = raíz, 0 = sin pesos.

    Returns
    -------
    torch.Tensor or None
        Tensor float de forma ``(n_columnas, n_clases)``, o ``None`` si ``power == 0``.
        Una clase que no aparece en ``dataset`` queda con peso 1: nunca es la etiqueta
        real, así que no cambia la loss.
    """
    if power == 0:
        return None
    counts = label_counts(dataset).double()
    present = counts > 0
    totals = counts.sum(dim=1, keepdim=True)
    n_present = present.sum(dim=1, keepdim=True)
    weights = torch.where(present, (totals / (n_present * counts)) ** power, torch.ones_like(counts))
    for column, (column_counts, column_weights) in enumerate(zip(counts.long(), weights)):
        logger.info(
            "Pesos por clase, columna %d: tokens=%s pesos=%s",
            column,
            column_counts.tolist(),
            [round(w, 2) for w in column_weights.tolist()],
        )
    return weights.float()
