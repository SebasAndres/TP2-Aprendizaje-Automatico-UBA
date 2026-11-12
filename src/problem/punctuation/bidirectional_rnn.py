import torch
from torch import nn

from src.constants import RANDOM_SEED

EMBEDDING_DIM = 768
HIDDEN_DIM = 128

class PunctuationBidirectionalRNN(nn.Module):
    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        # 1. Activamos la lectura en ambas direcciones
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # 2. Multiplicamos la entrada por 2 para las dos cabezas de salida
        self.out_ini = nn.Linear(hidden_dim * 2, 2)
        self.out_fin = nn.Linear(hidden_dim * 2, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        
        logits_ini = self.out_ini(h)
        logits_fin = self.out_fin(h)
        
        pad_shape = (x.size(0), x.size(1), 2)
        pad = torch.full(pad_shape, float('-inf'), device=x.device)
        logits_ini_padded = torch.cat([logits_ini, pad], dim=-1)
        
        return torch.stack([logits_ini_padded, logits_fin], dim=2)


def create_bidirectional_rnn_model() -> PunctuationBidirectionalRNN:
    """Crea la RNN bidireccional de puntuación, sin entrenar"""
    torch.manual_seed(RANDOM_SEED)
    return PunctuationBidirectionalRNN()
