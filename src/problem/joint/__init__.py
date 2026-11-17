"""Redes conjuntas: predicen capitalización y puntuación (las 3 tareas) a la vez.

`JointRNN` y `JointLSTM` (`rnn.py`) son las básicas. Las variantes de abajo tienen la misma
salida, un archivo cada una.

Los embeddings que reciben son estáticos: cada token tiene siempre el mismo vector, sin
importar la oración. Todo el contexto lo tiene que aportar la red. Por eso todas leen en las
dos direcciones: el "?" final depende de cómo empieza la pregunta, y la mayúscula inicial de
si antes terminó una oración. Cada una cambia una idea respecto de `bidirectional_joint_lstm`:

- `ProjectedBiGRU` (`joint_projected_bigru`): primero procesa cada token solo, con una capa
  lineal + GELU (768 -> 256), y recién después lee la oración con una GRU, que tiene menos
  parámetros que una LSTM. Las cabezas son MLPs (lineal + GELU + lineal) y no una sola lineal.
- `ResidualStackedBiLSTM` (`joint_residual_stacked_bilstm`): dos capas de BiLSTM con dropout
  entre ellas (la primera ve patrones locales y la segunda los de toda la oración). Además,
  las cabezas reciben el token proyectado junto con el contexto, así que lo que dice el
  embedding de la palabra (nombre propio, sigla) les llega directo, sin pasar por la LSTM.
- `ConvBiLSTM` (`joint_conv_bilstm`): una convolución sobre los tokens (ventana de 3) arma
  primero una representación de cada token con sus vecinos, que suelen ser las otras
  piezas de la misma palabra ("pre", "##gun", "##ta"). Después la BiLSTM lee la oración.
"""

from src.problem.joint.conv_bilstm import create_conv_bilstm_model
from src.problem.joint.projected_bigru import create_projected_bigru_model
from src.problem.joint.residual_stacked_bilstm import create_residual_stacked_bilstm_model
from src.problem.joint.rnn import create_joint_lstm_model, create_joint_model, create_joint_rnn_model, joint_model_name

__all__ = [
    "create_joint_rnn_model",
    "create_joint_lstm_model",
    "create_joint_model",
    "joint_model_name",
    "create_projected_bigru_model",
    "create_residual_stacked_bilstm_model",
    "create_conv_bilstm_model",
]
