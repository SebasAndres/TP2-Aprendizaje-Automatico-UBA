import torch
from torch import nn

from src.constants import RANDOM_SEED
from src.metrics.capitalization import N_CAPITALIZATION_CLASSES

EMBEDDING_DIM = 768  # tamaño de los embeddings estáticos de BERT
HIDDEN_DIM = 128     # tamaño del estado oculto h_t (hiperparámetro)


class CapitalizationRNN(nn.Module):
    """RNN estándar (unidireccional): para cada token devuelve un score (logit) por clase.

    En cada token t calcula h_t = tanh(W_x · x_t + W_h · h_{t-1} + b) y luego
    logits_t = W_y · h_t + b_y. Los mismos pesos se usan en todas las posiciones.
    No aplica softmax: lo hace la cross entropy de la métrica.
    """

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        n_classes: int = N_CAPITALIZATION_CLASSES,
    ):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)  # W_x, W_h, b
        self.out = nn.Linear(hidden_dim, n_classes)                  # W_y, b_y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_tokens, input_dim) -> logits: (batch, n_tokens, n_classes)"""
        h, _ = self.rnn(x)  # h: (batch, n_tokens, hidden_dim), un h_t por token
        return self.out(h)


def create_standard_rnn_model() -> CapitalizationRNN:
    """Crea la RNN estándar de capitalización, sin entrenar"""
    torch.manual_seed(RANDOM_SEED)  # misma inicialización de pesos en cada corrida
    return CapitalizationRNN()
