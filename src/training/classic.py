import logging
from typing import Any

from torch.utils.data import DataLoader
import numpy as np
import torch

from src.problem import Problem
from src.features import TextFeatureExtractor
from src.metrics.base import WordLevelTask
from src.training.result import TrainResult
from src.words import concat_word_ids, token_word_ids

logger = logging.getLogger(__name__)

LOG_EVERY_N_BATCHES = 50


def _batch_to_numpy(
    batch_X: list[list[str]],
    batch_y: torch.Tensor,
    feature_extractor: TextFeatureExtractor
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Devuelve las features, las etiquetas sin padding y las palabras de cada oración del batch"""
    sentences = []
    for tokens, padded_labels in zip(batch_X, batch_y):
        if not tokens:
            continue
        features = feature_extractor.extract(tokens)
        # Las etiquetas vienen con padding hasta el largo máximo del batch
        sentences.append((features.numpy(), padded_labels[:len(tokens)].numpy(), token_word_ids(tokens)))
    return sentences

def _dataloader_to_numpy(
    dataloader: DataLoader,
    feature_extractor: TextFeatureExtractor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recibe un dataloder del dataset y devuelve las matrices de numpy X, Y a nivel token.
    Omite las oraciones sin tokens.

    Returns
    -------
    - X: np.ndarray
        Matriz de features del dataset (n_tokens, n_features)
    - Y: np.ndarray
        Matriz de targets del dataset (n_tokens,) o (n_tokens, n_targets)
    - word_ids: np.ndarray
        Índice de la palabra de cada token (n_tokens,)
    """

    sentences = []
    for i, (batch_X, batch_y) in enumerate(dataloader, start=1):
        sentences.extend(_batch_to_numpy(batch_X, batch_y, feature_extractor))
        if i % LOG_EVERY_N_BATCHES == 0:
            logger.info("Extrayendo features: batch %d/%d", i, len(dataloader))
    X = np.concatenate([features for features, _, _ in sentences], axis=0)
    Y = np.concatenate([labels for _, labels, _ in sentences], axis=0)
    word_ids = concat_word_ids([ids for _, _, ids in sentences])
    logger.info("Features: %d/%d oraciones, X=%s, Y=%s", len(sentences), len(dataloader.dataset), X.shape, Y.shape)
    return X, Y, word_ids

def train_classic_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    problem: Problem
) -> TrainResult:
    """Ajusta el modelo clásico del problema sobre train y devuelve sus predicciones sobre
    test a nivel palabra. No entrena por épocas, así que no tiene historial"""
    logger.info("Extrayendo features de train")
    X_train, Y_train, _ = _dataloader_to_numpy(train_loader, feature_extractor)
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
    X, Y, word_ids = _dataloader_to_numpy(loader, feature_extractor)
    return problem.metric.word_level(model.predict(X), Y, word_ids)
