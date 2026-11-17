"""`joint_conv_bilstm`: convolución sobre los tokens vecinos + BiLSTM. Ver `src/problem/joint/__init__.py`."""

import torch
from torch import nn

from src.constants import EMBEDDING_DIM, HIDDEN_DIM, RANDOM_SEED
from src.problem.joint.layers import DROPOUT, PROJECTION_DIM, JointHeads, run_recurrent


class ConvBiLSTM(nn.Module):
    """Convolución sobre los tokens (ventana de 3, GELU) -> BiLSTM -> cabezas lineales.

    Parameters
    ----------
    hidden_dim : int, default=HIDDEN_DIM
        Dimensión del estado oculto de cada dirección de la LSTM.
    kernel_size : int, default=3
        Cantidad de tokens vecinos que mira la convolución.

    Attributes
    ----------
    conv : torch.nn.Conv1d
        Convolución de ``EMBEDDING_DIM`` a ``PROJECTION_DIM`` canales.
    activation : torch.nn.Sequential
        GELU + dropout sobre la salida de la convolución.
    rnn : torch.nn.LSTM
        LSTM bidireccional.
    dropout : torch.nn.Dropout
        Dropout sobre la salida de la LSTM.
    heads : JointHeads
        Una cabeza lineal por tarea.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(EMBEDDING_DIM, PROJECTION_DIM, kernel_size, padding=kernel_size // 2)
        self.activation = nn.Sequential(nn.GELU(), nn.Dropout(DROPOUT))
        self.rnn = nn.LSTM(PROJECTION_DIM, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.heads = JointHeads(2 * hidden_dim)

    def forward(self, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        # Conv1d espera (batch, canales, n_tokens). El padding de ceros del final actúa como
        # borde, y lo que la convolución calcula en esas posiciones la LSTM lo saltea
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
        tokens = self.activation(self.conv(x.transpose(1, 2)).transpose(1, 2))
        h = run_recurrent(self.rnn, tokens, lengths)
        return self.heads(self.dropout(h))


def create_conv_bilstm_model() -> ConvBiLSTM:
    """Crea la red ``ConvBiLSTM``, sin entrenar, con la semilla fija.

    Returns
    -------
    ConvBiLSTM
        Red con las dimensiones por defecto.
    """
    torch.manual_seed(RANDOM_SEED)
    return ConvBiLSTM()
