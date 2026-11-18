"""Cabeza de salida de las RNN bidireccionales: lineal o con una capa oculta no lineal."""

from torch import nn


def output_head(in_dim: int, n_classes: int, head_dim: int | None = None, dropout: float = 0.0) -> nn.Module:
    """Crea la cabeza que pasa del estado de la RNN a los logits de cada clase.

    Parameters
    ----------
    in_dim : int
        Dimensión de entrada (``2 * hidden_dim`` en las bidireccionales).
    n_classes : int
        Cantidad de clases de la tarea.
    head_dim : int or None, default=None
        Tamaño de la capa oculta. Con ``None`` la cabeza es una sola ``nn.Linear``
        (la arquitectura original); con un entero es
        ``Linear(in_dim, head_dim) -> GELU -> Dropout -> Linear(head_dim, n_classes)``.
    dropout : float, default=0.0
        Probabilidad de dropout entre las dos capas (solo si hay capa oculta).

    Returns
    -------
    torch.nn.Module
        Una ``nn.Linear`` o un ``nn.Sequential``.
    """
    if head_dim is None:
        return nn.Linear(in_dim, n_classes)
    return nn.Sequential(
        nn.Linear(in_dim, head_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_dim, n_classes)
    )
