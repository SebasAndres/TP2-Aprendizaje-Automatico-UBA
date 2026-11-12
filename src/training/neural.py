"""Entrenamiento genérico de redes neuronales que clasifican cada token.

El loop NO depende de la arquitectura: sirve para cualquier `nn.Module` (RNN estándar,
RNN bidireccional, LSTM, CNN, Transformer, un MLP por token, etc.) que cumpla este contrato:

- Entrada:  `forward(X, lengths)`. `X` de forma (n_oraciones, max_tokens, n_features), con
            padding de ceros; `lengths` es la cantidad de tokens reales de cada oración
            (sirve para que una RNN saltee el padding con pack_padded_sequence).
- Salida:   logits de forma (n_oraciones, max_tokens, ..., n_clases), compatibles con
            `problem.metric.loss` y con `argmax(dim=-1)` para predecir.

Para agregar una red nueva:
1. Implementarla en `src/problem/<problema>/` como un `nn.Module`.
2. Exponerla con un método en `Problem` (ej. `cnn_model`).
3. Registrar el experimento: `neural_trainer(lambda problem: problem.cnn_model())`.
"""

import logging
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.features import TextFeatureExtractor
from src.metrics.base import PerClass
from src.problem import Problem
from src.words import concat_word_ids, token_word_ids

logger = logging.getLogger(__name__)

N_EPOCHS = 5
LEARNING_RATE = 1e-3
LOG_EVERY_N_BATCHES = 50

ModelFactory = Callable[[Problem], nn.Module]
"""Crea una red nueva, sin entrenar, para el problema dado"""


class EpochMetrics(NamedTuple):
    """Métricas sobre train de una época: loss promedio, score del problema y
    (precision, recall) por clase"""
    epoch: int
    loss: float
    score: dict[str, float]
    per_class: PerClass


TrainResult = tuple[Any, dict[str, float], list[EpochMetrics]]
"""(modelo, score sobre test, métricas de train por época)"""

Trainer = Callable[[DataLoader, DataLoader, TextFeatureExtractor, Problem], TrainResult]
"""Firma que espera `Experiment.trainer`"""


class PredictionLog:
    """Acumula predicciones y etiquetas por token, sin padding, para calcular métricas"""

    def __init__(self) -> None:
        self.preds: list[np.ndarray] = []
        self.reals: list[np.ndarray] = []
        self.word_ids: list[np.ndarray] = []

    def add(self, pred: torch.Tensor, Y: torch.Tensor, tokens: list[list[str]]) -> None:
        """pred, Y: (n_oraciones, max_tokens, ...) en CPU"""
        for i, sentence in enumerate(tokens):
            n_tokens = len(sentence)
            self.preds.append(pred[i, :n_tokens].numpy())
            self.reals.append(Y[i, :n_tokens].numpy())
            self.word_ids.append(token_word_ids(sentence))

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.concatenate(self.preds), np.concatenate(self.reals), concat_word_ids(self.word_ids)

    def score(self, problem: Problem) -> dict[str, float]:
        return problem.metric.score(*self._arrays())

    def per_class(self, problem: Problem) -> PerClass:
        return problem.metric.per_class(*self._arrays())


def format_score(score: dict[str, float]) -> str:
    """Ej: 'punt_inicial=0.912 punt_final=0.655'"""
    return " ".join(f"{task}={value:.3f}" for task, value in score.items())


def batch_to_tensors(
    batch_X: list[list[str]],
    batch_y: torch.Tensor,
    feature_extractor: TextFeatureExtractor
) -> tuple[torch.Tensor, torch.Tensor, list[list[str]]] | None:
    """Extrae las features de cada oración del batch y las rellena con ceros hasta el largo
    máximo. Omite las oraciones sin tokens; devuelve None si no queda ninguna.

    Returns
    -------
    - X: torch.Tensor
        Features (n_oraciones, max_tokens, n_features)
    - Y: torch.Tensor
        Etiquetas (n_oraciones, max_tokens, ...), con IGNORE_INDEX en el padding
    - tokens: list[list[str]]
        Los tokens de cada oración
    """
    keep = [i for i, tokens in enumerate(batch_X) if tokens]
    if not keep:
        return None
    tokens = [batch_X[i] for i in keep]
    X = pad_sequence([feature_extractor.extract(sentence) for sentence in tokens], batch_first=True)
    Y = batch_y[keep, :X.size(1)]
    return X, Y, tokens


def sentence_lengths(tokens: list[list[str]]) -> list[int]:
    """Cantidad de tokens reales de cada oración (sin padding)"""
    return [len(sentence) for sentence in tokens]


def predict(model: nn.Module, X: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    """Clase más probable de cada token.

    X: (n_oraciones, max_tokens, n_features), en el mismo device que el modelo
    lengths: tokens reales de cada oración
    -> (n_oraciones, max_tokens, ...), incluye las posiciones de padding
    """
    model.eval()  # modo evaluación: Dropout apagado, BatchNorm usa sus estadísticas guardadas
    with torch.no_grad():
        return model(X, lengths).argmax(dim=-1)


def _evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    problem: Problem,
    device: torch.device
) -> dict[str, float]:
    """Predice sobre test, saca el padding de cada oración y calcula el score del problema"""
    log = PredictionLog()
    for batch_X, batch_y in test_loader:
        batch = batch_to_tensors(batch_X, batch_y, feature_extractor)
        if batch is None:
            continue
        X, Y, tokens = batch
        log.add(predict(model, X.to(device), sentence_lengths(tokens)).cpu(), Y, tokens)
    return log.score(problem)


def _train_step(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    tokens: list[list[str]],
    problem: Problem,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple[float, torch.Tensor]:
    """Actualiza los pesos con un batch. Devuelve la loss y la clase predicha de cada token (en CPU)"""
    logits = model(X.to(device), sentence_lengths(tokens))  # (n_oraciones, max_tokens, ..., n_clases)
    loss = problem.metric.loss(logits, Y.to(device))  # ignora el padding
    optimizer.zero_grad()  # borra los gradientes del batch anterior
    loss.backward()        # calcula los gradientes (en una RNN: backpropagation through time)
    optimizer.step()       # actualiza todos los parámetros de la red
    return loss.item(), logits.detach().argmax(dim=-1).cpu()


def _log_progress(i: int, n_batches: int, loss: float, log: PredictionLog, problem: Problem) -> None:
    """Loguea la loss del batch y el F1 de train acumulado en lo que va de la época.
    La precision/recall por clase queda en el gráfico de la época"""
    logger.info("  batch %d/%d  loss=%.4f  %s", i, n_batches, loss, format_score(log.score(problem)))


def _train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    problem: Problem,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple[float, PredictionLog]:
    """Hace una pasada completa sobre train. Devuelve la loss promedio por batch y las
    predicciones que hizo la red sobre train durante la época"""
    # No entrena: solo pone la red en modo entrenamiento (Dropout activo, BatchNorm con
    # estadísticas del batch). Va en cada época porque predict la deja en modo eval.
    model.train()
    log = PredictionLog()
    total_loss, n_batches = 0.0, 0
    for i, (batch_X, batch_y) in enumerate(train_loader, start=1):
        batch = batch_to_tensors(batch_X, batch_y, feature_extractor)
        if batch is None:
            continue
        X, Y, tokens = batch
        loss, pred = _train_step(model, X, Y, tokens, problem, optimizer, device)
        log.add(pred, Y, tokens)
        total_loss += loss
        n_batches += 1
        if i % LOG_EVERY_N_BATCHES == 0:
            _log_progress(i, len(train_loader), loss, log, problem)
    return total_loss / max(n_batches, 1), log


def train_neural_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    problem: Problem,
    build_model: ModelFactory
) -> TrainResult:
    """Crea la red con `build_model`, la entrena N_EPOCHS épocas y devuelve su score en test
    y las métricas sobre train de cada época. Si el problema no implementa esa red,
    devuelve (None, {}, [])."""
    try:
        model = build_model(problem)
    except NotImplementedError as error:
        logger.warning("Red sin implementar para %s: %s", type(problem).__name__, error)
        return None, {}, []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    logger.info("Entrenando %s en %s (%d parámetros)",
                type(model).__name__, device, sum(p.numel() for p in model.parameters()))

    history = []
    for epoch in range(1, N_EPOCHS + 1):
        avg_loss, log = _train_epoch(model, train_loader, feature_extractor, problem, optimizer, device)
        metrics = EpochMetrics(epoch, avg_loss, log.score(problem), log.per_class(problem))
        history.append(metrics)
        logger.info("Época %d/%d  loss=%.4f  %s", epoch, N_EPOCHS, avg_loss, format_score(metrics.score))

    logger.info("Prediciendo y calculando score sobre test")
    score = _evaluate(model, test_loader, feature_extractor, problem, device)
    return model.cpu(), score, history


def neural_trainer(build_model: ModelFactory) -> Trainer:
    """Adapta `train_neural_model` a la firma de `Experiment.trainer`, fijando qué red entrenar"""
    def trainer(
        train_loader: DataLoader,
        test_loader: DataLoader,
        feature_extractor: TextFeatureExtractor,
        problem: Problem
    ) -> TrainResult:
        return train_neural_model(train_loader, test_loader, feature_extractor, problem, build_model)
    return trainer
