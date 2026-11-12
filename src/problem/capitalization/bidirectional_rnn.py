import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.constants import RANDOM_SEED
from src.data.capitalization_dataset import N_CAPITALIZATION_CLASSES

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

    def forward(self, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        # pack/pad: la RNN saltea el padding (en la bidireccional, la lectura hacia atrás
        # empezaría por los ceros de relleno)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        return self.out(h)

def create_bidirectional_rnn_model() -> CapitalizationBidirectionalRNN:
    """Crea la RNN bidireccional de capitalización, sin entrenar"""
    torch.manual_seed(RANDOM_SEED)
    return CapitalizationBidirectionalRNN()
