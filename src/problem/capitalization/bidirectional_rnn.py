"""RNN bidireccional para el problema de capitalización."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.constants import EMBEDDING_DIM, HIDDEN_DIM, RANDOM_SEED
from src.data.capitalization_dataset import N_CAPITALIZATION_CLASSES


class CapitalizationBidirectionalRNN(nn.Module):
    """RNN bidireccional que clasifica la capitalización de cada token.

    Lee la oración en las dos direcciones: ``h→_t`` resume los tokens ``0..t`` y
    ``h←_t`` los tokens ``t..n-1``. La salida de cada token usa la concatenación
    ``[h→_t ; h←_t]``, así que conoce el contexto de toda la oración.

    Parameters
    ----------
    input_dim : int, default=EMBEDDING_DIM
        Dimensión del embedding de cada token (768 para BERT multilingüe).
    hidden_dim : int, default=HIDDEN_DIM
        Dimensión del estado oculto de cada dirección.
    n_classes : int, default=N_CAPITALIZATION_CLASSES
        Cantidad de clases de capitalización: 0 = minúscula, 1 = Inicial,
        2 = MiXta, 3 = MAYÚS.

    Attributes
    ----------
    rnn : torch.nn.RNN
        Capa recurrente bidireccional.
    out : torch.nn.Linear
        Capa de salida de ``2 * hidden_dim`` a ``n_classes``.

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
        # 1. Activamos la lectura en ambas direcciones
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # 2. Multiplicamos la entrada por 2 porque ahora concatena ambas memorias
        self.out = nn.Linear(hidden_dim * 2, n_classes)

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
            Logits por token y clase. Las posiciones de padding no tienen
            significado y la métrica las ignora.
        """
        # pack/pad: la RNN saltea el padding (en la bidireccional, la lectura hacia atrás
        # empezaría por los ceros de relleno)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        return self.out(h)

def create_bidirectional_rnn_model() -> CapitalizationBidirectionalRNN:
    """Crea la RNN bidireccional de capitalización, sin entrenar.

    Fija la semilla antes de construirla para que todas las corridas arranquen
    con los mismos pesos.

    Returns
    -------
    CapitalizationBidirectionalRNN
        Red con las dimensiones por defecto de ``src.constants``.

    Examples
    --------
    >>> model = create_bidirectional_rnn_model()
    >>> model.rnn.bidirectional
    True
    """
    torch.manual_seed(RANDOM_SEED)
    return CapitalizationBidirectionalRNN()
