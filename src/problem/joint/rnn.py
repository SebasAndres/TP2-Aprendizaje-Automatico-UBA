"""Una sola RNN que predice capitalización y puntuación al mismo tiempo.

La RNN se comparte entre las tres tareas y cada tarea tiene su propia "cabeza" lineal:
- capitalización: 4 clases (0, 1, 2, 3)
- puntuación inicial: 2 clases ("", ¿)
- puntuación final: 4 clases ("", ",", ".", "?")

`JointLSTM` es la misma red con celdas LSTM en lugar de la RNN simple: cada celda tiene
compuertas que deciden qué recordar y qué olvidar, así que conserva mejor el contexto lejano
(ej: el "¿" del principio de una pregunta larga para poner el "?" al final).

En el pipeline son los experimentos `joint_rnn`, `bidirectional_joint_rnn`, `joint_lstm` y
`bidirectional_joint_lstm`. Para entrenarlas solas ver `src/joint_rnn.py`.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.constants import EMBEDDING_DIM, HIDDEN_DIM, RANDOM_SEED
from src.data.capitalization_dataset import N_CAPITALIZATION_CLASSES
from src.data.punctuation_dataset import FINAL_LABELS, INITIAL_LABELS
from src.problem.joint.layers import pad_classes


class JointRNN(nn.Module):
    """RNN compartida + una cabeza lineal por tarea: capitalización y puntuación a la vez.

    La RNN calcula el estado oculto ``h_t`` de cada token y tres cabezas lineales lo leen:

    - capitalización: 4 clases (minúscula, Inicial, MiXta, MAYÚS)
    - puntuación inicial: 2 clases (``""``, ``"¿"``)
    - puntuación final: 4 clases (``""``, ``","``, ``"."``, ``"?"``)

    Parameters
    ----------
    input_dim : int, default=EMBEDDING_DIM
        Dimensión del embedding de cada token (768 para BERT multilingüe).
    hidden_dim : int, default=HIDDEN_DIM
        Dimensión del estado oculto de cada dirección.
    bidirectional : bool, default=False
        Si es ``True`` lee la oración en las dos direcciones y las cabezas reciben
        ``[h→_t ; h←_t]``, de dimensión ``2 * hidden_dim``.

    Attributes
    ----------
    rnn : torch.nn.RNN or torch.nn.LSTM
        Capa recurrente compartida, creada por ``_recurrent_layer``.
    capitalization_head : torch.nn.Linear
        Cabeza de capitalización, a ``N_CAPITALIZATION_CLASSES`` clases.
    initial_head : torch.nn.Linear
        Cabeza de puntuación inicial, a ``len(INITIAL_LABELS)`` clases.
    final_head : torch.nn.Linear
        Cabeza de puntuación final, a ``len(FINAL_LABELS)`` clases.

    See Also
    --------
    JointLSTM : la misma red con celdas LSTM.
    JointDataset : dataset con las 3 etiquetas por token que espera esta red.

    Notes
    -----
    Las tareas con menos clases se completan con ``-inf`` hasta ``max_clases`` (ver
    ``pad_classes``) para devolver un único tensor. Devuelve logits, sin softmax.
    """

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        # Parte compartida: lee la oración y construye el estado oculto h_t de cada token
        self.rnn = self._recurrent_layer(input_dim, hidden_dim, bidirectional)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        # Una cabeza por tarea, todas leen el mismo h_t
        self.capitalization_head = nn.Linear(out_dim, N_CAPITALIZATION_CLASSES)
        self.initial_head = nn.Linear(out_dim, len(INITIAL_LABELS))
        self.final_head = nn.Linear(out_dim, len(FINAL_LABELS))

    @staticmethod
    def _recurrent_layer(input_dim: int, hidden_dim: int, bidirectional: bool) -> nn.Module:
        """Crea la capa recurrente compartida: una RNN simple.

        Parameters
        ----------
        input_dim : int
            Dimensión del embedding de cada token.
        hidden_dim : int
            Dimensión del estado oculto de cada dirección.
        bidirectional : bool
            Si lee la instancia en las dos direcciones.

        Returns
        -------
        torch.nn.Module
            Una ``nn.RNN`` con ``batch_first=True``.
        """
        return nn.RNN(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)

    def forward(self, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        """Calcula los logits de las 3 tareas para cada token.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, n_tokens, input_dim)
            Embeddings de las oraciones, con padding al final hasta ``n_tokens``.
        lengths : list[int]
            Cantidad de tokens reales de cada oración (``len(lengths) == batch``).

        Returns
        -------
        torch.Tensor of shape (batch, n_tokens, 3, max_clases)
            Logits por token, tarea y clase. Las tareas van en el orden
            capitalización, puntuación inicial, puntuación final, y las clases que
            no existen en una tarea valen ``-inf``.
        """
        # pack/pad: la RNN saltea el padding (en la bidireccional, la lectura hacia atrás
        # empezaría por los ceros de relleno)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)  # la LSTM devuelve (h_n, c_n) en lugar de h_n: se descarta igual
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        heads = [self.capitalization_head(h), self.initial_head(h), self.final_head(h)]
        return torch.stack(pad_classes(heads), dim=2)


class JointLSTM(JointRNN):
    """La misma red conjunta de ``JointRNN``, con celdas LSTM.

    Además del estado oculto ``h_t`` lleva una memoria ``c_t`` que solo cambia a
    través de compuertas (olvidar, escribir, leer), así que conserva mejor el
    contexto lejano. Con el mismo ``hidden_dim`` tiene ~4 veces los parámetros
    recurrentes de la RNN simple.

    Parameters
    ----------
    input_dim : int, default=EMBEDDING_DIM
        Dimensión del embedding de cada token.
    hidden_dim : int, default=HIDDEN_DIM
        Dimensión del estado oculto (y de la memoria) de cada dirección.
    bidirectional : bool, default=False
        Si es ``True`` lee la oración en las dos direcciones.

    See Also
    --------
    JointRNN : misma interfaz y misma salida.
    """

    @staticmethod
    def _recurrent_layer(input_dim: int, hidden_dim: int, bidirectional: bool) -> nn.Module:
        """Crea la capa recurrente compartida: una LSTM.

        Parameters
        ----------
        input_dim : int
            Dimensión del embedding de cada token.
        hidden_dim : int
            Dimensión del estado oculto de cada dirección.
        bidirectional : bool
            Si lee la instancia en las dos direcciones.

        Returns
        -------
        torch.nn.Module
            Una ``nn.LSTM`` con ``batch_first=True``.
        """
        return nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)


def create_joint_rnn_model(bidirectional: bool = False) -> JointRNN:
    """Crea la RNN conjunta, sin entrenar, con la semilla fija.

    Parameters
    ----------
    bidirectional : bool, default=False
        Si la RNN lee la oración en las dos direcciones.

    Returns
    -------
    JointRNN
        Red con las dimensiones por defecto de ``src.constants``.
    """
    torch.manual_seed(RANDOM_SEED)  # misma inicialización de pesos en cada corrida
    return JointRNN(bidirectional=bidirectional)


def create_joint_lstm_model(bidirectional: bool = False) -> JointLSTM:
    """Crea la LSTM conjunta, sin entrenar, con la semilla fija.

    Parameters
    ----------
    bidirectional : bool, default=False
        Si la LSTM lee la oración en las dos direcciones.

    Returns
    -------
    JointLSTM
        Red con las dimensiones por defecto de ``src.constants``.
    """
    torch.manual_seed(RANDOM_SEED)
    return JointLSTM(bidirectional=bidirectional)


def create_joint_model(bidirectional: bool = False, lstm: bool = False) -> JointRNN:
    """Crea la red conjunta pedida, sin entrenar.

    Parameters
    ----------
    bidirectional : bool, default=False
        Si la red lee la oración en las dos direcciones.
    lstm : bool, default=False
        Si usa celdas LSTM (``JointLSTM``) en lugar de la RNN simple (``JointRNN``).

    Returns
    -------
    JointRNN
        Una ``JointRNN`` o una ``JointLSTM`` (que hereda de ``JointRNN``).
    """
    return create_joint_lstm_model(bidirectional) if lstm else create_joint_rnn_model(bidirectional)


def joint_model_name(bidirectional: bool = False, lstm: bool = False) -> str:
    """Arma el nombre del experimento, con el que se guarda el archivo.

    Parameters
    ----------
    bidirectional : bool, default=False
        Si la red lee en las dos direcciones.
    lstm : bool, default=False
        Si usa celdas LSTM.

    Returns
    -------
    str
        Ej: ``'bidirectional_joint_lstm'``.
    """
    return ("bidirectional_" if bidirectional else "") + ("joint_lstm" if lstm else "joint_rnn")
