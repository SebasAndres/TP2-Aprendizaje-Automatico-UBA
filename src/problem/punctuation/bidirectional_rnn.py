"""RNN bidireccional para el problema de puntuación."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.constants import EMBEDDING_DIM, HIDDEN_DIM, RANDOM_SEED


class PunctuationBidirectionalRNN(nn.Module):
    """RNN bidireccional que predice la puntuación inicial y final de cada token.

    Lee la oración en las dos direcciones y cada token usa ``[h→_t ; h←_t]``, así
    que conoce el contexto de toda la oración (ej: para poner ``"¿"`` al principio
    sabe si la oración termina en pregunta). Dos cabezas lineales leen ese estado:

    - puntuación inicial: 2 clases (``""``, ``"¿"``)
    - puntuación final: 4 clases (``""``, ``","``, ``"."``, ``"?"``)

    Parameters
    ----------
    input_dim : int, default=EMBEDDING_DIM
        Dimensión del embedding de cada token (768 para BERT multilingüe).
    hidden_dim : int, default=HIDDEN_DIM
        Dimensión del estado oculto de cada dirección.
    dropout : float, default=0.0
        Probabilidad de apagar cada valor de ``[h→_t ; h←_t]`` antes de las cabezas,
        solo durante el entrenamiento (regularización). Con 0 no se apaga nada.

    Attributes
    ----------
    rnn : torch.nn.RNN
        Capa recurrente bidireccional compartida por las dos tareas.
    out_ini : torch.nn.Linear
        Cabeza de puntuación inicial, de ``2 * hidden_dim`` a 2.
    out_fin : torch.nn.Linear
        Cabeza de puntuación final, de ``2 * hidden_dim`` a 4.

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
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # 1. Activamos la lectura en ambas direcciones
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        # El dropout= de nn.RNN solo actúa entre capas apiladas: con una sola capa no haría
        # nada, así que va como capa aparte sobre la salida de la RNN
        self.dropout = nn.Dropout(dropout)

        # 2. Multiplicamos la entrada por 2 para las dos cabezas de salida
        self.out_ini = nn.Linear(hidden_dim * 2, 2)
        self.out_fin = nn.Linear(hidden_dim * 2, 4)

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
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        h = self.dropout(h)  # en model.eval() no apaga nada

        logits_ini = self.out_ini(h)
        logits_fin = self.out_fin(h)

        pad_shape = (x.size(0), x.size(1), 2)
        pad = torch.full(pad_shape, float("-inf"), device=x.device)
        logits_ini_padded = torch.cat([logits_ini, pad], dim=-1)

        return torch.stack([logits_ini_padded, logits_fin], dim=2)


def create_bidirectional_rnn_model(dropout: float = 0.0) -> PunctuationBidirectionalRNN:
    """Crea la RNN bidireccional de puntuación, sin entrenar.

    Fija la semilla antes de construirla para que todas las corridas arranquen
    con los mismos pesos.

    Parameters
    ----------
    dropout : float, default=0.0
        Probabilidad de dropout sobre la salida de la RNN (0 = sin dropout).

    Returns
    -------
    PunctuationBidirectionalRNN
        Red con las dimensiones por defecto de ``src.constants``.
    """
    torch.manual_seed(RANDOM_SEED)
    return PunctuationBidirectionalRNN(dropout=dropout)
