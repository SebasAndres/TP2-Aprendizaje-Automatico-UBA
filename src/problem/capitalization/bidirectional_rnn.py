import torch
from torch import nn

from src.constants import RANDOM_SEED
from src.metrics.capitalization import N_CAPITALIZATION_CLASSES

EMBEDDING_DIM = 768
HIDDEN_DIM = 128

class CapitalizationBidirectionalRNN(nn.Module):
    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        n_classes: int = N_CAPITALIZATION_CLASSES,
    ):
        super().__init__()
        # 1. Activamos la lectura en ambas direcciones
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # 2. Multiplicamos la entrada por 2 porque ahora concatena ambas memorias
        self.out = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x) 
        return self.out(h)

def create_bidirectional_rnn_model() -> CapitalizationBidirectionalRNN:
    """Crea la RNN bidireccional de capitalización, sin entrenar"""
    torch.manual_seed(RANDOM_SEED)
    return CapitalizationBidirectionalRNN()
