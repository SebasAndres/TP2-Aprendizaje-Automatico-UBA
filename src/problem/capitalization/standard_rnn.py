"""RNN estándar (unidireccional) para el problema de capitalización."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.constants import EMBEDDING_DIM, HIDDEN_DIM, RANDOM_SEED
from src.data.capitalization_dataset import N_CAPITALIZATION_CLASSES


class CapitalizationRNN(nn.Module):
    """RNN estándar (unidireccional) que clasifica la capitalización de cada token.

    En cada token t calcula ``h_t = tanh(W_x · x_t + W_h · h_{t-1} + b)`` y luego
    ``logits_t = W_y · h_t + b_y``. Los mismos pesos se usan en todas las posiciones.
    Como lee de izquierda a derecha, ``h_t`` solo conoce los tokens ``0..t``.

    Parameters
    ----------
    input_dim : int, default=EMBEDDING_DIM
        Dimensión del embedding de cada token (768 para BERT multilingüe).
    hidden_dim : int, default=HIDDEN_DIM
        Dimensión del estado oculto ``h_t``.
    n_classes : int, default=N_CAPITALIZATION_CLASSES
        Cantidad de clases de capitalización: 0 = minúscula, 1 = Inicial,
        2 = MiXta, 3 = MAYÚS.

    Attributes
    ----------
    rnn : torch.nn.RNN
        Capa recurrente (``W_x``, ``W_h``, ``b``).
    out : torch.nn.Linear
        Capa de salida (``W_y``, ``b_y``) de ``hidden_dim`` a ``n_classes``.

    Notes
    -----
    Devuelve logits, sin softmax: la ``CrossEntropyLoss`` de la métrica lo aplica.
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

    def forward(self, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        """Calcula los logits de capitalización de cada token.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, n_tokens, input_dim)
            Embeddings de las oraciones, con padding al final hasta ``n_tokens``.
        lengths : list[int]
            Cantidad de tokens reales de cada oración (``len(lengths) == batch``).

        Returns
        -------
        torch.Tensor of shape (batch, n_tokens, n_classes)
            Logits por token y clase. En las posiciones de padding valen ``out(0)``
            (solo el bias) y la métrica las ignora.
        """
        # pack/pad: la RNN saltea el padding (en la bidireccional, la lectura hacia atrás
        # empezaría por los ceros de relleno)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))  # h: (batch, n_tokens, hidden_dim), un h_t por token
        return self.out(h)


def create_standard_rnn_model() -> CapitalizationRNN:
    """Crea la RNN estándar de capitalización, sin entrenar.

    Fija la semilla antes de construirla para que todas las corridas arranquen
    con los mismos pesos.

    Returns
    -------
    CapitalizationRNN
        Red con las dimensiones por defecto de ``src.constants``.

    Examples
    --------
    >>> model = create_standard_rnn_model()
    >>> model.out.out_features
    4
    """
    torch.manual_seed(RANDOM_SEED)  # misma inicialización de pesos en cada corrida
    return CapitalizationRNN()
