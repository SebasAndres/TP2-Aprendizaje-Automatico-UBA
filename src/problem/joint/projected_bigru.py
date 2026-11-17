"""`joint_projected_bigru`: proyección por token + BiGRU + cabezas MLP. Ver `src/problem/joint/__init__.py`."""

import torch
from torch import nn

from src.constants import HIDDEN_DIM, RANDOM_SEED
from src.problem.joint.layers import DROPOUT, PROJECTION_DIM, JointHeads, run_recurrent, token_projection


class ProjectedBiGRU(nn.Module):
    """Proyección por token (768 -> 256, GELU) -> BiGRU -> cabezas MLP.

    Parameters
    ----------
    hidden_dim : int, default=HIDDEN_DIM
        Dimensión del estado oculto de cada dirección de la GRU y de la capa oculta
        de cada cabeza.

    Attributes
    ----------
    projection : torch.nn.Sequential
        Capa por token (ver ``token_projection``).
    rnn : torch.nn.GRU
        GRU bidireccional.
    dropout : torch.nn.Dropout
        Dropout sobre la salida de la GRU.
    heads : JointHeads
        Una cabeza MLP por tarea.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.projection = token_projection()
        self.rnn = nn.GRU(PROJECTION_DIM, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.heads = JointHeads(2 * hidden_dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        """Calcula los logits de las 3 tareas para cada token.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, n_tokens, EMBEDDING_DIM)
            Embeddings de las instancias, con padding al final hasta ``n_tokens``.
        lengths : list[int]
            Cantidad de tokens reales de cada instancia (``len(lengths) == batch``).

        Returns
        -------
        torch.Tensor of shape (batch, n_tokens, 3, max_clases)
            Logits por token, tarea y clase, como ``JointRNN.forward``.
        """
        h = run_recurrent(self.rnn, self.projection(x), lengths)
        return self.heads(self.dropout(h))


def create_projected_bigru_model() -> ProjectedBiGRU:
    """Crea la red ``ProjectedBiGRU``, sin entrenar, con la semilla fija.

    Returns
    -------
    ProjectedBiGRU
        Red con las dimensiones por defecto.
    """
    torch.manual_seed(RANDOM_SEED)
    return ProjectedBiGRU()
