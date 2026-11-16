"""Modelos clásicos (no secuenciales), entrenados a nivel palabra.

Cada fila es una palabra: sus features (`extract_words` del extractor) y sus etiquetas
(`word_labels` de la métrica: la capitalización de la palabra, la puntuación inicial de su
primer token y la final del último). Así una palabra de varios tokens es una sola fila, como
en la evaluación, y no hay filas con las mismas features y distinta etiqueta (con el
promedio de embeddings, todos los tokens de `cu ##ánd ##o?` tenían la misma X, pero solo el
último tenía el "?"). Al predecir, la clase de cada palabra se copia a todos sus tokens.
"""
import logging
from typing import Any

from torch.utils.data import DataLoader
import numpy as np
import torch

from src.problem import Problem
from src.features import TextFeatureExtractor
from src.metrics import Metric
from src.metrics.base import WordLevelTask
from src.training.result import TrainResult
from src.words import token_word_ids

logger = logging.getLogger(__name__)

LOG_EVERY_N_BATCHES = 50


def _batch_to_numpy(
    batch_X: list[list[str]],
    batch_y: torch.Tensor,
    feature_extractor: TextFeatureExtractor,
    metric: Metric
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Devuelve las features y las etiquetas por palabra de cada oración del batch"""
    sentences = []
    for tokens, padded_labels in zip(batch_X, batch_y):
        if not tokens:
            continue
        features = feature_extractor.extract_words(tokens).numpy()
        # Las etiquetas vienen con padding hasta el largo máximo del batch
        labels = metric.word_labels(padded_labels[:len(tokens)].numpy(), token_word_ids(tokens))
        sentences.append((features, labels))
    return sentences


def _dataloader_to_numpy(
    dataloader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    metric: Metric
) -> tuple[np.ndarray, np.ndarray]:
    """Recibe un dataloader del dataset y devuelve las matrices de numpy X, Y a nivel palabra.
    Omite las oraciones sin tokens.

    Returns
    -------
    - X: np.ndarray
        Matriz de features del dataset (n_palabras, n_features)
    - Y: np.ndarray
        Matriz de targets del dataset (n_palabras,) o (n_palabras, n_targets)
    """
    sentences = []
    for i, (batch_X, batch_y) in enumerate(dataloader, start=1):
        sentences.extend(_batch_to_numpy(batch_X, batch_y, feature_extractor, metric))
        if i % LOG_EVERY_N_BATCHES == 0:
            logger.info("Extrayendo features: batch %d/%d", i, len(dataloader))
    X = np.concatenate([features for features, _ in sentences], axis=0)
    Y = np.concatenate([labels for _, labels in sentences], axis=0)
    logger.info("Features: %d/%d oraciones, X=%s, Y=%s", len(sentences), len(dataloader.dataset), X.shape, Y.shape)
    return X, Y


def train_classic_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    problem: Problem
) -> TrainResult:
    """Ajusta el modelo clásico del problema sobre train y devuelve sus predicciones sobre
    test a nivel palabra. No entrena por épocas, así que no tiene historial"""
    logger.info("Extrayendo features de train")
    X_train, Y_train = _dataloader_to_numpy(train_loader, feature_extractor, problem.metric)
    model = problem.classic_model()
    logger.info("Ajustando %s", type(model).__name__)
    model.fit(X_train, Y_train)

    logger.info("Prediciendo sobre test")
    return TrainResult(model, evaluate_classic_model(model, test_loader, feature_extractor, problem))


def evaluate_classic_model(
    model: Any,
    loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    problem: Problem
) -> dict[str, WordLevelTask]:
    """Predice con un modelo clásico ya ajustado y devuelve las clases reales y predichas
    a nivel palabra de cada tarea"""
    X, Y = _dataloader_to_numpy(loader, feature_extractor, problem.metric)
    # Cada fila ya es una palabra: cada "token" es su propia palabra
    return problem.metric.word_level(model.predict(X), Y, np.arange(len(Y)))
