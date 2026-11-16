"""Variantes de la red conjunta (las 3 tareas a la vez, misma salida que `JointRNN`).

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
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.constants import EMBEDDING_DIM, HIDDEN_DIM, RANDOM_SEED
from src.data.capitalization_dataset import N_CAPITALIZATION_CLASSES
from src.data.punctuation_dataset import FINAL_LABELS, INITIAL_LABELS
from src.joint_rnn import _pad_classes

PROJECTION_DIM = 256  # tamaño de cada token después de la capa por token (proyección o convolución)
DROPOUT = 0.3         # probabilidad de apagar cada valor durante el entrenamiento
TASK_CLASSES = (N_CAPITALIZATION_CLASSES, len(INITIAL_LABELS), len(FINAL_LABELS))


def run_recurrent(rnn: nn.Module, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    """Pasa `x` (batch, n_tokens, dim) por la capa recurrente salteando el padding"""
    packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    h, _ = rnn(packed)
    h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
    return h


def token_projection(input_dim: int = EMBEDDING_DIM) -> nn.Sequential:
    """Procesa cada token por separado: lineal + GELU + dropout"""
    return nn.Sequential(nn.Linear(input_dim, PROJECTION_DIM), nn.GELU(), nn.Dropout(DROPOUT))


class JointHeads(nn.Module):
    """Una cabeza por tarea (capitalización, puntuación inicial, final) sobre el mismo h_t.
    Con `hidden_dim` cada cabeza es un MLP (lineal + GELU + dropout + lineal); sin él, una lineal"""

    def __init__(self, in_dim: int, hidden_dim: int | None = None):
        super().__init__()
        self.heads = nn.ModuleList(self._head(in_dim, n_classes, hidden_dim) for n_classes in TASK_CLASSES)

    @staticmethod
    def _head(in_dim: int, n_classes: int, hidden_dim: int | None) -> nn.Module:
        if hidden_dim is None:
            return nn.Linear(in_dim, n_classes)
        return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(DROPOUT),
                             nn.Linear(hidden_dim, n_classes))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(batch, n_tokens, in_dim) -> logits (batch, n_tokens, 3, max_clases), como JointRNN"""
        return torch.stack(_pad_classes([head(h) for head in self.heads]), dim=2)


class ProjectedBiGRU(nn.Module):
    """Proyección por token (768 -> 256, GELU) -> BiGRU -> cabezas MLP"""

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.projection = token_projection()
        self.rnn = nn.GRU(PROJECTION_DIM, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.heads = JointHeads(2 * hidden_dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        h = run_recurrent(self.rnn, self.projection(x), lengths)
        return self.heads(self.dropout(h))


class ResidualStackedBiLSTM(nn.Module):
    """Proyección por token -> BiLSTM de 2 capas -> cabezas que leen [contexto ; token proyectado]"""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_layers: int = 2):
        super().__init__()
        self.projection = token_projection()
        self.rnn = nn.LSTM(PROJECTION_DIM, hidden_dim, num_layers=num_layers, dropout=DROPOUT,
                           batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.heads = JointHeads(2 * hidden_dim + PROJECTION_DIM)

    def forward(self, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        tokens = self.projection(x)
        h = run_recurrent(self.rnn, tokens, lengths)
        return self.heads(self.dropout(torch.cat([h, tokens], dim=-1)))


class ConvBiLSTM(nn.Module):
    """Convolución sobre los tokens (ventana de 3, GELU) -> BiLSTM -> cabezas lineales"""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(EMBEDDING_DIM, PROJECTION_DIM, kernel_size, padding=kernel_size // 2)
        self.activation = nn.Sequential(nn.GELU(), nn.Dropout(DROPOUT))
        self.rnn = nn.LSTM(PROJECTION_DIM, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.heads = JointHeads(2 * hidden_dim)

    def forward(self, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        # Conv1d espera (batch, canales, n_tokens). El padding de ceros del final actúa como
        # borde, y lo que la convolución calcula en esas posiciones la LSTM lo saltea
        tokens = self.activation(self.conv(x.transpose(1, 2)).transpose(1, 2))
        h = run_recurrent(self.rnn, tokens, lengths)
        return self.heads(self.dropout(h))


def create_projected_bigru_model() -> ProjectedBiGRU:
    torch.manual_seed(RANDOM_SEED)
    return ProjectedBiGRU()


def create_residual_stacked_bilstm_model() -> ResidualStackedBiLSTM:
    torch.manual_seed(RANDOM_SEED)
    return ResidualStackedBiLSTM()


def create_conv_bilstm_model() -> ConvBiLSTM:
    torch.manual_seed(RANDOM_SEED)
    return ConvBiLSTM()
