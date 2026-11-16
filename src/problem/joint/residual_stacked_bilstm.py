"""`joint_residual_stacked_bilstm`: BiLSTM de 2 capas y cabezas que también ven el token proyectado. Ver `src/problem/joint/__init__.py`."""

import torch
from torch import nn

from src.constants import HIDDEN_DIM, RANDOM_SEED
from src.problem.joint.layers import DROPOUT, PROJECTION_DIM, JointHeads, run_recurrent, token_projection


class ResidualStackedBiLSTM(nn.Module):
    """Proyección por token -> BiLSTM de 2 capas -> cabezas que leen [contexto ; token proyectado]"""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_layers: int = 2):
        super().__init__()
        self.projection = token_projection()
        self.rnn = nn.LSTM(
            PROJECTION_DIM, hidden_dim, num_layers=num_layers, dropout=DROPOUT, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.heads = JointHeads(2 * hidden_dim + PROJECTION_DIM)

    def forward(self, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        tokens = self.projection(x)
        h = run_recurrent(self.rnn, tokens, lengths)
        return self.heads(self.dropout(torch.cat([h, tokens], dim=-1)))


def create_residual_stacked_bilstm_model() -> ResidualStackedBiLSTM:
    torch.manual_seed(RANDOM_SEED)
    return ResidualStackedBiLSTM()
