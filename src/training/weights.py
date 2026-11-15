"""Pesos por clase para la CrossEntropy.

La mayoría de los tokens no lleva puntuación ni mayúscula, así que una red que siempre
predice "nada"/"minúscula" ya tiene una loss baja. Con pesos, equivocarse en una clase
rara cuesta más que equivocarse en una común.

El peso de la clase c de una tarea es

    w_c = (N / (K · n_c)) ** power

con N los tokens de la tarea, K las clases que aparecen y n_c los tokens de la clase c.
- power = 1: "balanced", cada clase pesa lo mismo en total. Con clases muy raras
  (¿ es ~1% de los tokens, MAYÚS ~0.1%) los pesos llegan a 50-250 y la red tiende a
  predecir la clase rara de más.
- power = 0.5: raíz de lo anterior, un punto medio (pesos de ~0.5 a ~15).
- power = 0: sin pesos.
"""
import logging

import torch
from torch.utils.data import Dataset

from src.data import IGNORE_INDEX

logger = logging.getLogger(__name__)


def label_counts(dataset: Dataset) -> torch.Tensor:
    """Cantidad de tokens de cada clase en cada columna de etiquetas, sin contar el padding.
    -> (n_columnas, n_clases). Una tarea con menos clases que otra queda con ceros al final"""
    labels = torch.cat([y.reshape(len(y), -1) for _, y in dataset])  # (n_tokens, n_columnas)
    n_classes = int(labels.max()) + 1
    return torch.stack([
        torch.bincount(column[column != IGNORE_INDEX], minlength=n_classes)
        for column in labels.T
    ])


def class_weights(dataset: Dataset, power: float) -> torch.Tensor | None:
    """Pesos de cada clase de cada columna de etiquetas, calculados sobre `dataset` (el de
    train). -> (n_columnas, n_clases), o None si power es 0 (sin pesos). Una clase que no
    aparece en `dataset` queda con peso 1: nunca es la etiqueta real, así que no cambia la loss"""
    if power == 0:
        return None
    counts = label_counts(dataset).double()
    present = counts > 0
    totals = counts.sum(dim=1, keepdim=True)
    n_present = present.sum(dim=1, keepdim=True)
    weights = torch.where(present, (totals / (n_present * counts)) ** power, torch.ones_like(counts))
    for column, (column_counts, column_weights) in enumerate(zip(counts.long(), weights)):
        logger.info("Pesos por clase, columna %d: tokens=%s pesos=%s", column,
                    column_counts.tolist(), [round(w, 2) for w in column_weights.tolist()])
    return weights.float()
