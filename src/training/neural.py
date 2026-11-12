"""Entrenamiento genérico de redes neuronales que clasifican cada token.

El loop NO depende de la arquitectura: sirve para cualquier `nn.Module` (RNN estándar,
RNN bidireccional, LSTM, CNN, Transformer, un MLP por token, etc.) que cumpla este contrato:

- Entrada:  `X` de forma (n_oraciones, max_tokens, n_features), con padding de ceros.
- Salida:   logits de forma (n_oraciones, max_tokens, ..., n_clases), compatibles con
            `problem.metric.loss` y con `argmax(dim=-1)` para predecir.

Para agregar una red nueva:
1. Implementarla en `src/problem/<problema>/` como un `nn.Module`.
2. Exponerla con un método en `Problem` (ej. `cnn_model`).
3. Registrar el experimento: `neural_trainer(lambda problem: problem.cnn_model())`.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.features import TextFeatureExtractor
from src.problem import Problem
from src.words import concat_word_ids, token_word_ids

logger = logging.getLogger(__name__)

N_EPOCHS = 5
LEARNING_RATE = 1e-3
LOG_EVERY_N_BATCHES = 50

ModelFactory = Callable[[Problem], nn.Module]
"""Crea una red nueva, sin entrenar, para el problema dado"""

Trainer = Callable[[DataLoader, DataLoader, TextFeatureExtractor, Problem], tuple[Any, dict[str, float]]]
"""Firma que espera `Experiment.trainer`"""


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


def predict(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Clase más probable de cada token.

    X: (n_oraciones, max_tokens, n_features), en el mismo device que el modelo
    -> (n_oraciones, max_tokens, ...), incluye las posiciones de padding
    """
    model.eval()  # modo evaluación: Dropout apagado, BatchNorm usa sus estadísticas guardadas
    with torch.no_grad():
        return model(X).argmax(dim=-1)


def _evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    problem: Problem,
    device: torch.device
) -> dict[str, float]:
    """Predice sobre test, saca el padding de cada oración y calcula el score del problema"""
    preds, reals, word_ids = [], [], []
    for batch_X, batch_y in test_loader:
        batch = batch_to_tensors(batch_X, batch_y, feature_extractor)
        if batch is None:
            continue
        X, Y, tokens = batch
        pred = predict(model, X.to(device)).cpu()
        for i, sentence in enumerate(tokens):
            n_tokens = len(sentence)
            preds.append(pred[i, :n_tokens].numpy())
            reals.append(Y[i, :n_tokens].numpy())
            word_ids.append(token_word_ids(sentence))
    return problem.metric.score(np.concatenate(preds), np.concatenate(reals), concat_word_ids(word_ids))


def _train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    problem: Problem,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Hace una pasada completa sobre train y devuelve la loss promedio por batch"""
    # No entrena: solo pone la red en modo entrenamiento (Dropout activo, BatchNorm con
    # estadísticas del batch). Va en cada época porque predict la deja en modo eval.

    model.train()
    total_loss, n_batches = 0.0, 0
    for i, (batch_X, batch_y) in enumerate(train_loader, start=1):
        batch = batch_to_tensors(batch_X, batch_y, feature_extractor)
        if batch is None:
            continue
        X, Y, _ = batch

        logits = model(X.to(device))                     # (n_oraciones, max_tokens, ..., n_clases)
        loss = problem.metric.loss(logits, Y.to(device))  # ignora el padding
        optimizer.zero_grad()  # borra los gradientes del batch anterior
        loss.backward()        # calcula los gradientes (en una RNN: backpropagation through time)
        optimizer.step()       # actualiza todos los parámetros de la red

        total_loss += loss.item()
        n_batches += 1
        if i % LOG_EVERY_N_BATCHES == 0:
            logger.info("Batch %d/%d, loss=%.4f", i, len(train_loader), loss.item())
    return total_loss / max(n_batches, 1)


def train_neural_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    problem: Problem,
    build_model: ModelFactory
) -> tuple[nn.Module | None, dict[str, float]]:
    """Crea la red con `build_model`, la entrena N_EPOCHS épocas y devuelve su score en test.
    Si el problema no implementa esa red, devuelve (None, {})."""
    try:
        model = build_model(problem)
    except NotImplementedError as error:
        logger.warning("Red sin implementar para %s: %s", type(problem).__name__, error)
        return None, {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Entrenando %s en %s (%d parámetros)", type(model).__name__, device, n_params)

    for epoch in range(1, N_EPOCHS + 1):
        avg_loss = _train_epoch(model, train_loader, feature_extractor, problem, optimizer, device)
        logger.info("Época %d/%d terminada: loss promedio=%.4f", epoch, N_EPOCHS, avg_loss)

    logger.info("Prediciendo y calculando score sobre test")
    score = _evaluate(model, test_loader, feature_extractor, problem, device)
    return model.cpu(), score


def neural_trainer(build_model: ModelFactory) -> Trainer:
    """Adapta `train_neural_model` a la firma de `Experiment.trainer`, fijando qué red entrenar"""
    def trainer(
        train_loader: DataLoader,
        test_loader: DataLoader,
        feature_extractor: TextFeatureExtractor,
        problem: Problem
    ) -> tuple[nn.Module | None, dict[str, float]]:
        return train_neural_model(train_loader, test_loader, feature_extractor, problem, build_model)
    return trainer
