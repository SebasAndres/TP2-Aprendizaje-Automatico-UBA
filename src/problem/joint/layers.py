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
    """Pasa `x` (batch, n_tokens, dim) por la capa recurrente salteando el padding"""
    packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    h, _ = rnn(packed)
    h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
    return h


def token_projection(input_dim: int = EMBEDDING_DIM) -> nn.Sequential:
    """Procesa cada token por separado: lineal + GELU + dropout"""
    return nn.Sequential(nn.Linear(input_dim, PROJECTION_DIM), nn.GELU(), nn.Dropout(DROPOUT))


class JointHeads(nn.Module):
    """Una cabeza por tarea (capitalización, puntuación inicial, final) sobre el mismo h_t.
    Con `hidden_dim` cada cabeza es un MLP (lineal + GELU + dropout + lineal); sin él, una lineal"""

    def __init__(self, in_dim: int, hidden_dim: int | None = None):
        super().__init__()
        self.heads = nn.ModuleList(self._head(in_dim, n_classes, hidden_dim) for n_classes in TASK_CLASSES)

    @staticmethod
    def _head(in_dim: int, n_classes: int, hidden_dim: int | None) -> nn.Module:
        if hidden_dim is None:
            return nn.Linear(in_dim, n_classes)
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(batch, n_tokens, in_dim) -> logits (batch, n_tokens, 3, max_clases), como `JointRNN`"""
        return torch.stack(pad_classes([head(h) for head in self.heads]), dim=2)
