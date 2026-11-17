"""Piezas que comparten las redes conjuntas: cabezas por tarea, proyección por token y
el relleno de clases para apilar las 3 tareas en un tensor."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.constants import EMBEDDING_DIM
from src.data.capitalization_dataset import N_CAPITALIZATION_CLASSES
from src.data.punctuation_dataset import FINAL_LABELS, INITIAL_LABELS

PROJECTION_DIM = 256  # tamaño de cada token después de la capa por token (proyección o convolución)
DROPOUT = 0.3  # probabilidad de apagar cada valor durante el entrenamiento
TASK_CLASSES = (N_CAPITALIZATION_CLASSES, len(INITIAL_LABELS), len(FINAL_LABELS))


def pad_classes(logits: list[torch.Tensor]) -> list[torch.Tensor]:
    """Completa con ``-inf`` las tareas con menos clases para poder apilarlas.

    Con ``-inf`` la clase inventada tiene probabilidad 0: nunca se predice ni cambia la loss.

    Parameters
    ----------
    logits : list[torch.Tensor]
        Logits de cada tarea, de shape ``(..., n_classes_tarea)``. Todas las
        dimensiones salvo la última deben coincidir.

    Returns
    -------
    list[torch.Tensor]
        Los mismos logits, todos de shape ``(..., max_clases)``.

    Examples
    --------
    >>> padded = pad_classes([torch.zeros(1, 2), torch.zeros(1, 4)])
    >>> [task.shape for task in padded]
    [torch.Size([1, 4]), torch.Size([1, 4])]
    >>> padded[0]
    tensor([[0., 0., -inf, -inf]])
    """
    n_classes = max(task.size(-1) for task in logits)
    return [nn.functional.pad(task, (0, n_classes - task.size(-1)), value=float("-inf")) for task in logits]


def run_recurrent(rnn: nn.Module, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    """Pasa ``x`` por la capa recurrente salteando el padding.

    Parameters
    ----------
    rnn : torch.nn.Module
        Capa recurrente con ``batch_first=True`` (RNN, GRU o LSTM).
    x : torch.Tensor of shape (batch, n_tokens, dim)
        Entrada, con padding al final hasta ``n_tokens``.
    lengths : list[int]
        Cantidad de tokens reales de cada instancia.

    Returns
    -------
    torch.Tensor of shape (batch, n_tokens, hidden)
        Estado oculto de cada token, con ceros en el padding.
    """
    packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    h, _ = rnn(packed)
    h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
    return h


def token_projection(input_dim: int = EMBEDDING_DIM) -> nn.Sequential:
    """Crea la capa que procesa cada token por separado: lineal + GELU + dropout.

    Parameters
    ----------
    input_dim : int, default=EMBEDDING_DIM
        Dimensión de cada token de entrada.

    Returns
    -------
    torch.nn.Sequential
        Capa de ``input_dim`` a ``PROJECTION_DIM``.
    """
    return nn.Sequential(nn.Linear(input_dim, PROJECTION_DIM), nn.GELU(), nn.Dropout(DROPOUT))


class JointHeads(nn.Module):
    """Una cabeza por tarea (capitalización, puntuación inicial, final) sobre el mismo ``h_t``.

    Parameters
    ----------
    in_dim : int
        Dimensión de la entrada de cada cabeza.
    hidden_dim : int or None, default=None
        Con un valor, cada cabeza es un MLP (lineal + GELU + dropout + lineal) con esa
        capa oculta; con ``None``, una sola lineal.

    Attributes
    ----------
    heads : torch.nn.ModuleList
        Las 3 cabezas, en el orden de ``TASK_CLASSES``.
    """

    def __init__(self, in_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.heads = nn.ModuleList(self._head(in_dim, n_classes, hidden_dim) for n_classes in TASK_CLASSES)

    @staticmethod
    def _head(in_dim: int, n_classes: int, hidden_dim: int | None) -> nn.Module:
        """Crea la cabeza de una tarea.

        Parameters
        ----------
        in_dim : int
            Dimensión de la entrada.
        n_classes : int
            Cantidad de clases de la tarea.
        hidden_dim : int or None
            Capa oculta del MLP, o ``None`` para una sola lineal.

        Returns
        -------
        torch.nn.Module
            Una ``nn.Linear`` o un ``nn.Sequential``.
        """
        if hidden_dim is None:
            return nn.Linear(in_dim, n_classes)
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Aplica las 3 cabezas y apila sus logits.

        Parameters
        ----------
        h : torch.Tensor of shape (batch, n_tokens, in_dim)
            Representación de cada token.

        Returns
        -------
        torch.Tensor of shape (batch, n_tokens, 3, max_clases)
            Logits por tarea, como ``JointRNN``; las clases que no existen valen ``-inf``.
        """
        return torch.stack(pad_classes([head(h) for head in self.heads]), dim=2)
