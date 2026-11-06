from typing import Any

from torch.utils.data import DataLoader
import numpy as np

from src.problem import Problem
from src.features import TextFeatureExtractor


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
    for batch_X, batch_y in dataloader:
        for tokens, padded_labels in zip(batch_X, batch_y):
            features = feature_extractor.extract(tokens) if tokens else None
            if features is None:
                continue
            X_list.append(features.numpy())
            # Las etiquetas vienen con padding hasta el largo máximo del batch
            y_list.append(padded_labels[:len(tokens)].numpy())
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(y_list, axis=0)
    return X, Y

def train_classic_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    problem: Problem
) -> Any:

    X_train, Y_train = _dataloader_to_numpy(train_loader, feature_extractor)

    model = problem.classic_model()
    model.fit(X_train, Y_train)

    X_test, Y_test = _dataloader_to_numpy(test_loader, feature_extractor)

    pred_test = model.predict(X_test)
    score = problem.metric.score(pred_test, Y_test)

    print(score)
    return model
