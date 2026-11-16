"""RNN estándar (unidireccional) para el problema de puntuación."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.constants import EMBEDDING_DIM, HIDDEN_DIM, RANDOM_SEED


class PunctuationRNN(nn.Module):
    """RNN estándar (unidireccional) que predice la puntuación inicial y final de cada token.

    Una RNN compartida calcula el estado oculto ``h_t`` de cada token y dos cabezas
    lineales lo leen:

    - puntuación inicial: 2 clases (``""``, ``"¿"``)
    - puntuación final: 4 clases (``""``, ``","``, ``"."``, ``"?"``)

    Parameters
    ----------
    input_dim : int, default=EMBEDDING_DIM
        Dimensión del embedding de cada token (768 para BERT multilingüe).
    hidden_dim : int, default=HIDDEN_DIM
        Dimensión del estado oculto ``h_t``.

    Attributes
    ----------
    rnn : torch.nn.RNN
        Capa recurrente compartida por las dos tareas.
    out_ini : torch.nn.Linear
        Cabeza de puntuación inicial, de ``hidden_dim`` a 2.
    out_fin : torch.nn.Linear
        Cabeza de puntuación final, de ``hidden_dim`` a 4.

    Notes
    -----
    Para devolver un único tensor, los logits de la puntuación inicial se completan
    con ``-inf`` hasta 4 clases: esas clases inventadas tienen probabilidad 0, así
    que nunca se predicen ni cambian la loss. Devuelve logits, sin softmax.
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

    def forward(self, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        """Calcula los logits de puntuación inicial y final de cada token.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, n_tokens, input_dim)
            Embeddings de las oraciones, con padding al final hasta ``n_tokens``.
        lengths : list[int]
            Cantidad de tokens reales de cada oración (``len(lengths) == batch``).

        Returns
        -------
        torch.Tensor of shape (batch, n_tokens, 2, 4)
            Logits por token, tarea y clase. ``[..., 0, :]`` es la puntuación
            inicial (sus clases 2 y 3 valen ``-inf``) y ``[..., 1, :]`` la final.
        """
        # pack/pad: la RNN saltea el padding (en la bidireccional, la lectura hacia atrás
        # empezaría por los ceros de relleno)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))  # (batch, n_tokens, hidden_dim)
        
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
    """Crea la RNN estándar de puntuación, sin entrenar.

    Fija la semilla antes de construirla para que todas las corridas arranquen
    con los mismos pesos.

    Returns
    -------
    PunctuationRNN
        Red con las dimensiones por defecto de ``src.constants``.
    """
    torch.manual_seed(RANDOM_SEED)  # Misma inicialización para pruebas justas
    return PunctuationRNN()
