"""Una sola RNN que predice capitalización y puntuación al mismo tiempo.

La RNN se comparte entre las tres tareas y cada tarea tiene su propia "cabeza" lineal:
- capitalización: 4 clases (0, 1, 2, 3)
- puntuación inicial: 2 clases ("", ¿)
- puntuación final: 4 clases ("", ",", ".", "?")

En el pipeline es el experimento `joint_rnn`: se entrena con las 3 tareas sobre las mismas
oraciones de train que el resto de los modelos, y para cada problema se evalúan solo sus
cabezas. También se puede correr sola, con las 3 tareas a la vez:

    uv run python -m src.joint_rnn                      # dataset completo
    uv run python -m src.joint_rnn --sentences 30000    # solo las primeras 30000 oraciones
    uv run python -m src.joint_rnn --bidirectional      # RNN bidireccional
"""
import argparse
import logging
import sys

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, random_split

from src.constants import BATCH_SIZE, DATASET_PATH, RANDOM_SEED, TEST_RATIO
from src.data import CapitalizationDataset, PunctuationDataset, collate_batch, split_dataset_dev_control
from src.data.punctuation_dataset import FINAL_LABELS, INITIAL_LABELS
from src.features import EmbeddingsExtractor
from src.metrics import JointMetric
from src.metrics.capitalization import N_CAPITALIZATION_CLASSES
from src.problem.capitalization.standard_rnn import EMBEDDING_DIM, HIDDEN_DIM

logger = logging.getLogger("src.joint_rnn")


class JointDataset(CapitalizationDataset, PunctuationDataset):
    """Cada token tiene 3 etiquetas: <capitalización, puntuación inicial, puntuación final>"""

    def _expected_output(self, words: list[str], word_tokens: list[list[str]]) -> torch.Tensor:
        capitalization = CapitalizationDataset._expected_output(self, words, word_tokens)  # (n_tokens,)
        punctuation = PunctuationDataset._expected_output(self, words, word_tokens)        # (n_tokens, 2)
        return torch.cat([capitalization.unsqueeze(1), punctuation], dim=1)                # (n_tokens, 3)


class JointRNN(nn.Module):
    """RNN compartida + una cabeza lineal por tarea. Devuelve logits (sin softmax) por token"""

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        bidirectional: bool = False,
    ):
        super().__init__()
        # Parte compartida: lee la oración y construye el estado oculto h_t de cada token
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        # Una cabeza por tarea, todas leen el mismo h_t
        self.capitalization_head = nn.Linear(out_dim, N_CAPITALIZATION_CLASSES)
        self.initial_head = nn.Linear(out_dim, len(INITIAL_LABELS))
        self.final_head = nn.Linear(out_dim, len(FINAL_LABELS))

    def forward(self, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        """x: (batch, n_tokens, input_dim), lengths: tokens reales de cada oración
        -> logits: (batch, n_tokens, 3 tareas, max_clases)"""
        # pack/pad: la RNN saltea el padding (en la bidireccional, la lectura hacia atrás
        # empezaría por los ceros de relleno)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        heads = [self.capitalization_head(h), self.initial_head(h), self.final_head(h)]
        return torch.stack(_pad_classes(heads), dim=2)


def _pad_classes(logits: list[torch.Tensor]) -> list[torch.Tensor]:
    """Completa con -inf las tareas con menos clases para poder apilarlas. Con -inf la
    clase inventada tiene probabilidad 0: nunca se predice ni cambia la loss"""
    n_classes = max(task.size(-1) for task in logits)
    return [
        nn.functional.pad(task, (0, n_classes - task.size(-1)), value=float("-inf"))
        for task in logits
    ]


def create_joint_rnn_model(bidirectional: bool = False) -> JointRNN:
    """Crea la RNN conjunta, sin entrenar"""
    torch.manual_seed(RANDOM_SEED)  # misma inicialización de pesos en cada corrida
    return JointRNN(bidirectional=bidirectional)


def _loaders(dataset: Dataset) -> tuple[DataLoader, DataLoader]:
    """Mismos splits y semillas que el pipeline"""
    development_set, _ = split_dataset_dev_control(dataset)
    train_set, test_set = random_split(
        development_set,
        [1 - TEST_RATIO, TEST_RATIO],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    logger.info("Oraciones: train=%d, test=%d", len(train_set), len(test_set))
    return (
        DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch),
        DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch),
    )


def main() -> None:
    # Import local: src.training importa este módulo para el experimento joint_rnn
    from src.training.neural import N_EPOCHS, train_neural_model
    from src.training.result import format_score

    parser = argparse.ArgumentParser(description="Entrena una RNN conjunta de capitalización y puntuación")
    parser.add_argument("--sentences", type=int, default=None, help="usar solo las primeras N oraciones (default: todas)")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS, help=f"épocas de entrenamiento (default: {N_EPOCHS})")
    parser.add_argument("--bidirectional", action="store_true", help="usar una RNN bidireccional")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("src").setLevel(logging.INFO)

    dataset = JointDataset(DATASET_PATH, args.sentences or sys.maxsize)
    train_loader, test_loader = _loaders(dataset)
    model = create_joint_rnn_model(args.bidirectional)
    result = train_neural_model(
        train_loader, test_loader, EmbeddingsExtractor(), JointMetric(), model, n_epochs=args.epochs
    )
    print(f"\nJointRNN - F1 macro sobre test: {format_score(result.score)}")


if __name__ == "__main__":
    main()
