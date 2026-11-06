import logging

from torch.utils.data import DataLoader
import numpy as np
import torch

from src.problem import Problem
from src.features import TextFeatureExtractor

logger = logging.getLogger(__name__)

LOG_EVERY_N_BATCHES = 50


def _batch_to_numpy(
    batch_X: list[list[str]],
    batch_y: torch.Tensor,
    feature_extractor: TextFeatureExtractor
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extrae las features de cada oración del batch y recorta el padding de sus etiquetas"""
    X_list = []
    y_list = []
    for tokens, padded_labels in zip(batch_X, batch_y):
        features = feature_extractor.extract(tokens) if tokens else None
        if features is None:
            continue
        X_list.append(features.numpy())
        # Las etiquetas vienen con padding hasta el largo máximo del batch
        y_list.append(padded_labels[:len(tokens)].numpy())
    return X_list, y_list

def _dataloader_to_numpy(
    dataloader: DataLoader,
    feature_extractor: TextFeatureExtractor
) -> tuple[np.ndarray, np.ndarray]:
    """Recibe un dataloder del dataset y devuelve las matrices de numpy X, Y a nivel token.
    Omite las oraciones sin tokens y aquellas que el extractor no puede representar.

    Returns
    -------
    - X: np.ndarray
        Matriz de features del dataset (n_tokens, n_features)
    - Y: np.ndarray
        Matriz de targets del dataset (n_tokens,) o (n_tokens, n_targets)
    """

    X_list = []
    y_list = []
    for i, (batch_X, batch_y) in enumerate(dataloader, start=1):
        batch_X_list, batch_y_list = _batch_to_numpy(batch_X, batch_y, feature_extractor)
        X_list.extend(batch_X_list)
        y_list.extend(batch_y_list)
        if i % LOG_EVERY_N_BATCHES == 0:
            logger.info("Extrayendo features: batch %d/%d", i, len(dataloader))
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(y_list, axis=0)
    logger.info("Features: %d/%d oraciones, X=%s, Y=%s", len(X_list), len(dataloader.dataset), X.shape, Y.shape)
    return X, Y

def train_classic_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    problem: Problem
) -> tuple[any, float]:

    logger.info("Extrayendo features de train")
    X_train, Y_train = _dataloader_to_numpy(train_loader, feature_extractor)
    model = problem.classic_model()
    logger.info("Ajustando %s", type(model).__name__)
    model.fit(X_train, Y_train)

    logger.info("Extrayendo features de test")
    X_test, Y_test = _dataloader_to_numpy(test_loader, feature_extractor)
    logger.info("Prediciendo y calculando score sobre test")
    pred_test = model.predict(X_test)
    score = problem.metric.score(pred_test, Y_test)

    return model, score
