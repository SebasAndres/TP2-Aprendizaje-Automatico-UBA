"""`joint_residual_stacked_bilstm`: BiLSTM de 2 capas y cabezas que también ven el token proyectado. Ver `src/problem/joint/__init__.py`."""

import torch
from torch import nn

from src.constants import HIDDEN_DIM, RANDOM_SEED
from src.problem.joint.layers import DROPOUT, PROJECTION_DIM, JointHeads, run_recurrent, token_projection


class ResidualStackedBiLSTM(nn.Module):
    """Proyección por token -> BiLSTM de 2 capas -> cabezas que leen [contexto ; token proyectado].

    Parameters
    ----------
    hidden_dim : int, default=HIDDEN_DIM
        Dimensión del estado oculto de cada dirección de la LSTM.
    num_layers : int, default=2
        Cantidad de capas de LSTM apiladas, con dropout entre ellas.

    Attributes
    ----------
    projection : torch.nn.Sequential
        Capa por token (ver ``token_projection``).
    rnn : torch.nn.LSTM
        LSTM bidireccional apilada.
    dropout : torch.nn.Dropout
        Dropout sobre ``[h_t ; token proyectado]``.
    heads : JointHeads
        Una cabeza lineal por tarea.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_layers: int = 2) -> None:
        super().__init__()
        self.projection = token_projection()
        self.rnn = nn.LSTM(
            PROJECTION_DIM, hidden_dim, num_layers=num_layers, dropout=DROPOUT, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.heads = JointHeads(2 * hidden_dim + PROJECTION_DIM)

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
        tokens = self.projection(x)
        h = run_recurrent(self.rnn, tokens, lengths)
        return self.heads(self.dropout(torch.cat([h, tokens], dim=-1)))


def create_residual_stacked_bilstm_model() -> ResidualStackedBiLSTM:
    """Crea la red ``ResidualStackedBiLSTM``, sin entrenar, con la semilla fija.

    Returns
    -------
    ResidualStackedBiLSTM
        Red con las dimensiones por defecto.
    """
    torch.manual_seed(RANDOM_SEED)
    return ResidualStackedBiLSTM()
