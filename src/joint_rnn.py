"""Prototipo: una sola RNN que predice capitalización y puntuación al mismo tiempo.

La RNN se comparte entre las tres tareas y cada tarea tiene su propia "cabeza" lineal:
- capitalización: 4 clases (0, 1, 2, 3)
- puntuación inicial: 2 clases ("", ¿)
- puntuación final: 4 clases ("", ",", ".", "?")

Está fuera del pipeline a propósito, para discutir si conviene compartir pesos entre tareas.
Usa los mismos splits, semillas e hiperparámetros que la RNN estándar de capitalización,
así los scores se pueden comparar.

Uso:
    uv run python -m src.joint_rnn                      # dataset completo
    uv run python -m src.joint_rnn --sentences 30000    # solo las primeras 30000 oraciones
    uv run python -m src.joint_rnn --bidirectional      # RNN bidireccional
"""
import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Subset, random_split

from src.constants import BATCH_SIZE, DATASET_PATH, RANDOM_SEED, TEST_RATIO
from src.data import (
    IGNORE_INDEX,
    CapitalizationDataset,
    PunctuationDataset,
    collate_batch,
    split_dataset_dev_control,
)
from src.data.punctuation_dataset import FINAL_LABELS, INITIAL_LABELS
from src.features import EmbeddingsExtractor, TextFeatureExtractor
from src.metrics import CapitalizationMetric, PunctuationMetric
from src.metrics.capitalization import N_CAPITALIZATION_CLASSES
from src.problem.capitalization.standard_rnn import EMBEDDING_DIM, HIDDEN_DIM
from src.training.standard_rnn import LEARNING_RATE, N_EPOCHS, _batch_to_tensors
from src.words import concat_word_ids, token_word_ids

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

    def forward(
        self, x: torch.Tensor, lengths: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (batch, n_tokens, input_dim) -> logits de cada tarea: (batch, n_tokens, n_clases)"""
        # pack/pad: la RNN saltea el padding (en la bidireccional, la lectura hacia atrás
        # empezaría por los ceros de relleno)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        return self.capitalization_head(h), self.initial_head(h), self.final_head(h)


def _cross_entropy(logits: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """CrossEntropy ignorando el padding. Logits (..., n_clases), etiquetas (...)"""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        real.reshape(-1),
        ignore_index=IGNORE_INDEX
    )


def joint_loss(outputs: tuple[torch.Tensor, ...], Y: torch.Tensor) -> torch.Tensor:
    """Suma de la pérdida de cada tarea. Y: (batch, n_tokens, 3)"""
    capitalization, initial, final = outputs
    return (
        _cross_entropy(capitalization, Y[..., 0])
        + _cross_entropy(initial, Y[..., 1])
        + _cross_entropy(final, Y[..., 2])
    )


def _lengths(tokens: list[list[str]]) -> list[int]:
    return [len(sentence) for sentence in tokens]


def train(
    model: JointRNN,
    dataloader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    n_epochs: int,
    device: torch.device
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch_X, batch_y in dataloader:
            batch = _batch_to_tensors(batch_X, batch_y, feature_extractor)
            if batch is None:
                continue
            X, Y, tokens = batch

            outputs = model(X.to(device), _lengths(tokens))
            loss = joint_loss(outputs, Y.to(device))
            optimizer.zero_grad()
            loss.backward()   # los gradientes de las 3 tareas se suman en la RNN compartida
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        logger.info("Época %d/%d terminada: loss promedio=%.4f", epoch, n_epochs, total_loss / max(n_batches, 1))


def evaluate(
    model: JointRNN,
    dataloader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    device: torch.device
) -> dict[str, float]:
    """F1 macro a nivel palabra de cada tarea, con las mismas métricas que los otros modelos"""
    model.eval()
    preds, reals, word_ids = [], [], []
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch = _batch_to_tensors(batch_X, batch_y, feature_extractor)
            if batch is None:
                continue
            X, Y, tokens = batch
            outputs = model(X.to(device), _lengths(tokens))
            # Clase más probable de cada tarea: (n_oraciones, max_tokens, 3)
            pred = torch.stack([logits.argmax(dim=-1) for logits in outputs], dim=-1).cpu()
            for i, sentence in enumerate(tokens):
                n_tokens = len(sentence)
                preds.append(pred[i, :n_tokens].numpy())
                reals.append(Y[i, :n_tokens].numpy())
                word_ids.append(token_word_ids(sentence))

    pred, real = np.concatenate(preds), np.concatenate(reals)
    word_ids = concat_word_ids(word_ids)
    return {
        **CapitalizationMetric().score(pred[:, 0], real[:, 0], word_ids),
        **PunctuationMetric().score(pred[:, 1:], real[:, 1:], word_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena una RNN conjunta de capitalización y puntuación")
    parser.add_argument("--sentences", type=int, default=None, help="usar solo las primeras N oraciones (default: todas)")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS, help=f"épocas de entrenamiento (default: {N_EPOCHS})")
    parser.add_argument("--bidirectional", action="store_true", help="usar una RNN bidireccional")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("src").setLevel(logging.INFO)

    dataset = JointDataset(DATASET_PATH)
    if args.sentences is not None:
        dataset = Subset(dataset, range(min(args.sentences, len(dataset))))

    # Mismos splits y semillas que el pipeline
    development_set, _ = split_dataset_dev_control(dataset)
    train_set, test_set = random_split(
        development_set,
        [1 - TEST_RATIO, TEST_RATIO],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    logger.info("Oraciones: train=%d, test=%d", len(train_set), len(test_set))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    feature_extractor = EmbeddingsExtractor()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_SEED)
    model = JointRNN(bidirectional=args.bidirectional).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    kind = "bidireccional" if args.bidirectional else "estándar"
    logger.info("Entrenando JointRNN %s en %s (%d parámetros)", kind, device, n_params)

    train(model, train_loader, feature_extractor, args.epochs, device)
    logger.info("Prediciendo y calculando score sobre test")
    scores = evaluate(model, test_loader, feature_extractor, device)

    print(f"\nJointRNN {kind} - F1 macro sobre test:")
    for task, score in scores.items():
        print(f"  {task:15s} {score:.3f}")


if __name__ == "__main__":
    main()
