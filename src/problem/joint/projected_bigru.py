"""`joint_projected_bigru`: proyección por token + BiGRU + cabezas MLP. Ver `src/problem/joint/__init__.py`."""

import torch
from torch import nn

from src.constants import HIDDEN_DIM, RANDOM_SEED
from src.problem.joint.layers import DROPOUT, PROJECTION_DIM, JointHeads, run_recurrent, token_projection


class ProjectedBiGRU(nn.Module):
    """Proyección por token (768 -> 256, GELU) -> BiGRU -> cabezas MLP"""

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.projection = token_projection()
        self.rnn = nn.GRU(PROJECTION_DIM, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.heads = JointHeads(2 * hidden_dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        h = run_recurrent(self.rnn, self.projection(x), lengths)
        return self.heads(self.dropout(h))


def create_projected_bigru_model() -> ProjectedBiGRU:
    torch.manual_seed(RANDOM_SEED)
    return ProjectedBiGRU()
