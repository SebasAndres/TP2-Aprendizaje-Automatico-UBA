import torch
from torch import nn

from src.constants import RANDOM_SEED

EMBEDDING_DIM = 768  # tamaño de los embeddings estáticos de BERT
HIDDEN_DIM = 128     # tamaño del estado oculto h_t (hiperparámetro)

class PunctuationRNN(nn.Module):
    """
    RNN estándar para Puntuación. 
    Procesa la secuencia y devuelve logits para la puntuación inicial y final.
    """

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        # Tronco común de la red recurrente
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        
        # Cabeza para Puntuación Inicial (2 clases: "", "¿")
        self.out_ini = nn.Linear(hidden_dim, 2)
        
        # Cabeza para Puntuación Final (4 clases: "", ",", ".", "?")
        self.out_fin = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_tokens, input_dim) 
        -> logits: (batch, n_tokens, 2 tareas, 4 clases)
        """
        h, _ = self.rnn(x)  # (batch, n_tokens, hidden_dim)
        
        logits_ini = self.out_ini(h)  # (batch, n_tokens, 2)
        logits_fin = self.out_fin(h)  # (batch, n_tokens, 4)
        
        # Rellenamos logits_ini con -infinito para que tenga 4 columnas igual que out_fin.
        # Al usar -inf, nos aseguramos de que el modelo nunca prediga las clases inventadas.
        pad_shape = (x.size(0), x.size(1), 2)
        pad = torch.full(pad_shape, float('-inf'), device=x.device)
        logits_ini_padded = torch.cat([logits_ini, pad], dim=-1)
        
        # Apilamos las dos salidas en la dimensión 2
        return torch.stack([logits_ini_padded, logits_fin], dim=2)


def create_standard_rnn_model() -> PunctuationRNN:
    """Crea la RNN estándar de puntuación, sin entrenar"""
    torch.manual_seed(RANDOM_SEED)  # Misma inicialización para pruebas justas
    return PunctuationRNN()
