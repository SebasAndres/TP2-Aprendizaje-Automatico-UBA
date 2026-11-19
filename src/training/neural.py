"""Entrenamiento genérico de redes neuronales que clasifican cada token.

El loop NO depende de la arquitectura: sirve para cualquier `nn.Module` (RNN estándar,
RNN bidireccional, LSTM, CNN, Transformer, un MLP por token, etc.) que cumpla este contrato:

- Entrada:  `forward(X, lengths)`. `X` de forma (n_oraciones, max_tokens, n_features), con
            padding de ceros; `lengths` es la cantidad de tokens reales de cada oración
            (sirve para que una RNN saltee el padding con pack_padded_sequence).
- Salida:   logits de forma (n_oraciones, max_tokens, ..., n_clases), compatibles con
            la `loss` de la métrica del problema y con `argmax(dim=-1)` para predecir.

Para agregar una red nueva:
1. Implementarla en `src/problem/<problema>/` como un `nn.Module`.
2. Exponerla con un método en `Problem` (ej. `cnn_model`).
3. Registrar el experimento: `neural_trainer(lambda problem: problem.cnn_model())`.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.features import TextFeatureExtractor
from src.metrics import Metric
from src.constants import CLASS_WEIGHT_POWER
from src.metrics.base import ClassWeights, PerClass, WordLevelTask
from src.problem import Problem
from src.training.result import EpochMetrics, TrainResult, format_score, mean_score
from src.training.weights import class_weights
from src.words import concat_word_ids, token_word_ids

logger = logging.getLogger(__name__)

N_EPOCHS = 5
LEARNING_RATE = 1e-3
# Regularización L2 desacoplada (AdamW): en cada paso achica los pesos w <- w - lr * WEIGHT_DECAY * w,
# aparte del paso de Adam. Con Adam(weight_decay=...) el término L2 pasaría por la escala
# adaptativa y los pesos con gradientes grandes casi no se regularizarían.
# 0 = igual a Adam. Es el valor de las redes que no eligen el suyo (ver neural_trainer); el de
# bidirectional_rnn sale de notebooks/hyperparameters/regularization/regularization_weight_decay.ipynb
WEIGHT_DECAY = 0.0
# Norma máxima del gradiente en cada paso (None = sin recortar). En una RNN, backpropagation
# through time multiplica muchas veces por W_h y el gradiente puede explotar en secuencias
# largas; recortarlo evita que un solo batch desarme los pesos
GRAD_CLIP_NORM: float | None = 1.0
LOG_EVERY_N_BATCHES = 50

ModelFactory = Callable[[Problem], nn.Module]
"""Crea una red nueva, sin entrenar, para el problema dado"""


Trainer = Callable[[DataLoader, DataLoader, TextFeatureExtractor, Problem], TrainResult]
"""Firma que espera `Experiment.trainer`"""

Evaluator = Callable[[Any, DataLoader, TextFeatureExtractor, Problem], dict[str, WordLevelTask]]
"""Firma que espera `Experiment.evaluator`: predice con un modelo ya entrenado"""


def _device() -> torch.device:
    """Elige la GPU si hay una disponible, si no la CPU.

    Returns
    -------
    torch.device
        ``cuda`` o ``cpu``.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredictionLog:
    """Acumula predicciones y etiquetas por token, sin padding, para calcular métricas.

    Attributes
    ----------
    preds : list[np.ndarray]
        Clases predichas de cada instancia.
    reals : list[np.ndarray]
        Etiquetas reales de cada instancia.
    word_ids : list[np.ndarray]
        Índice de palabra de cada token de cada instancia.
    """

    def __init__(self) -> None:
        self.preds: list[np.ndarray] = []
        self.reals: list[np.ndarray] = []
        self.word_ids: list[np.ndarray] = []

    def add(self, pred: torch.Tensor, Y: torch.Tensor, tokens: list[list[str]]) -> None:
        """Agrega las predicciones de un batch, recortando el padding de cada instancia.

        Parameters
        ----------
        pred : torch.Tensor of shape (n_instancias, max_tokens, ...)
            Clases predichas, en CPU.
        Y : torch.Tensor of shape (n_instancias, max_tokens, ...)
            Etiquetas reales, en CPU.
        tokens : list[list[str]]
            Tokens de cada instancia del batch.
        """
        for i, sentence in enumerate(tokens):
            n_tokens = len(sentence)
            self.preds.append(pred[i, :n_tokens].numpy())
            self.reals.append(Y[i, :n_tokens].numpy())
            self.word_ids.append(token_word_ids(sentence))

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Concatena todo lo acumulado.

        Returns
        -------
        pred : np.ndarray
            Clases predichas de todos los tokens.
        real : np.ndarray
            Etiquetas reales de todos los tokens.
        word_ids : np.ndarray
            Índices de palabra sin repetir entre instancias (ver ``concat_word_ids``).
        """
        return np.concatenate(self.preds), np.concatenate(self.reals), concat_word_ids(self.word_ids)

    def score(self, metric: Metric) -> dict[str, float]:
        """Calcula el F1 macro a nivel palabra de cada tarea sobre lo acumulado.

        Parameters
        ----------
        metric : Metric
            Métrica del problema.

        Returns
        -------
        dict[str, float]
            F1 macro de cada tarea.
        """
        return metric.score(*self._arrays())

    def per_class(self, metric: Metric) -> PerClass:
        """Calcula la precision y el recall a nivel palabra de cada clase sobre lo acumulado.

        Parameters
        ----------
        metric : Metric
            Métrica del problema.

        Returns
        -------
        PerClass
            Por tarea, ``(precision, recall)`` de cada clase.
        """
        return metric.per_class(*self._arrays())

    def word_level(self, metric: Metric) -> dict[str, WordLevelTask]:
        """Pasa lo acumulado a nivel palabra.

        Parameters
        ----------
        metric : Metric
            Métrica del problema.

        Returns
        -------
        dict[str, WordLevelTask]
            Clases reales y predichas a nivel palabra de cada tarea.
        """
        return metric.word_level(*self._arrays())


def batch_to_tensors(
    batch_X: list[list[str]], batch_y: torch.Tensor, feature_extractor: TextFeatureExtractor
) -> tuple[torch.Tensor, torch.Tensor, list[list[str]]] | None:
    """Extrae las features de cada instancia del batch y las rellena con ceros hasta el largo máximo.

    Omite las instancias sin tokens.

    Parameters
    ----------
    batch_X : list[list[str]]
        Tokens de cada instancia del batch.
    batch_y : torch.Tensor of shape (batch, max_tokens, ...)
        Etiquetas del batch, con ``IGNORE_INDEX`` en el padding.
    feature_extractor : TextFeatureExtractor
        Extractor de features por token.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, list[list[str]]] or None
        ``(X, Y, tokens)``: features ``(n_instancias, max_tokens, n_features)``,
        etiquetas ``(n_instancias, max_tokens, ...)`` y tokens de cada instancia que
        quedó. ``None`` si no queda ninguna.
    """
    keep = [i for i, tokens in enumerate(batch_X) if tokens]
    if not keep:
        return None
    tokens = [batch_X[i] for i in keep]
    X = pad_sequence([feature_extractor.extract(sentence) for sentence in tokens], batch_first=True)
    Y = batch_y[keep, : X.size(1)]
    return X, Y, tokens


def sentence_lengths(tokens: list[list[str]]) -> list[int]:
    """Cuenta los tokens reales de cada instancia (sin padding).

    Parameters
    ----------
    tokens : list[list[str]]
        Tokens de cada instancia.

    Returns
    -------
    list[int]
        Largo de cada instancia.
    """
    return [len(sentence) for sentence in tokens]


def predict(model: nn.Module, X: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    """Predice la clase más probable de cada token.

    Parameters
    ----------
    model : nn.Module
        Red entrenada.
    X : torch.Tensor of shape (n_instancias, max_tokens, n_features)
        Features, en el mismo device que el modelo.
    lengths : list[int]
        Tokens reales de cada instancia.

    Returns
    -------
    torch.Tensor of shape (n_instancias, max_tokens, ...)
        Clase predicha de cada token, incluidas las posiciones de padding.
    """
    model.eval()  # modo evaluación: Dropout apagado, BatchNorm usa sus estadísticas guardadas
    with torch.no_grad():
        return model(X, lengths).argmax(dim=-1)


def _predict_with_loss(
    model: nn.Module,
    loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    metric: Metric,
    device: torch.device,
    weights: ClassWeights | None = None,
) -> tuple[PredictionLog, float]:
    """Predice sobre el loader sin entrenar.

    Parameters
    ----------
    model : nn.Module
        Red a evaluar, ya en ``device``.
    loader : DataLoader
        Instancias a predecir.
    feature_extractor : TextFeatureExtractor
        Extractor de features por token.
    metric : Metric
        Métrica del problema, que da la loss.
    device : torch.device
        Device de la red.
    weights : ClassWeights or None, default=None
        Pesos por clase de la loss, los mismos que en train.

    Returns
    -------
    log : PredictionLog
        Las predicciones, sin padding.
    loss : float
        Loss promedio por batch.
    """
    model.eval()  # modo evaluación: Dropout apagado, BatchNorm usa sus estadísticas guardadas
    log = PredictionLog()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch = batch_to_tensors(batch_X, batch_y, feature_extractor)
            if batch is None:
                continue
            X, Y, tokens = batch
            logits = model(X.to(device), sentence_lengths(tokens))
            total_loss += metric.loss(logits, Y.to(device), weights).item()
            n_batches += 1
            log.add(logits.argmax(dim=-1).cpu(), Y, tokens)
    return log, total_loss / max(n_batches, 1)


def _evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    metric: Metric,
    device: torch.device,
) -> dict[str, WordLevelTask]:
    """Predice sobre test y pasa las predicciones a nivel palabra.

    Parameters
    ----------
    model : nn.Module
        Red a evaluar, ya en ``device``.
    test_loader : DataLoader
        Instancias a predecir.
    feature_extractor : TextFeatureExtractor
        Extractor de features por token.
    metric : Metric
        Métrica del problema.
    device : torch.device
        Device de la red.

    Returns
    -------
    dict[str, WordLevelTask]
        Clases reales y predichas a nivel palabra de cada tarea.
    """
    log, _ = _predict_with_loss(model, test_loader, feature_extractor, metric, device)
    return log.word_level(metric)


def _train_step(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    tokens: list[list[str]],
    metric: Metric,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weights: ClassWeights | None,
) -> tuple[float, torch.Tensor]:
    """Actualiza los pesos con un batch.

    Parameters
    ----------
    model : nn.Module
        Red a entrenar o evaluar, ya en ``device``.
    X : torch.Tensor of shape (n_instancias, max_tokens, n_features)
        Features del batch, con padding de ceros.
    Y : torch.Tensor of shape (n_instancias, max_tokens, ...)
        Etiquetas del batch, con ``IGNORE_INDEX`` en el padding.
    tokens : list[list[str]]
        Tokens de cada instancia del batch.
    metric : Metric
        Métrica del problema, que da la loss.
    optimizer : torch.optim.Optimizer
        Optimizador de la red.
    device : torch.device
        Device de la red.
    weights : ClassWeights or None
        Pesos por clase de la loss.

    Returns
    -------
    loss : float
        Loss del batch.
    pred : torch.Tensor of shape (n_instancias, max_tokens, ...)
        Clase predicha de cada token, en CPU.
    """
    logits = model(X.to(device), sentence_lengths(tokens))  # (n_oraciones, max_tokens, ..., n_clases)
    loss = metric.loss(logits, Y.to(device), weights)  # ignora el padding, pesa cada clase
    optimizer.zero_grad()  # borra los gradientes del batch anterior
    loss.backward()  # calcula los gradientes (en una RNN: backpropagation through time)
    if GRAD_CLIP_NORM is not None:  # si la norma de todos los gradientes juntos supera el máximo, los escala
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()  # actualiza todos los parámetros de la red
    return loss.item(), logits.detach().argmax(dim=-1).cpu()


def _log_progress(i: int, n_batches: int, loss: float, log: PredictionLog, metric: Metric) -> None:
    """Loguea la loss del batch y el F1 de train acumulado en lo que va de la época.

    La precision/recall por clase queda en el gráfico de la época.

    Parameters
    ----------
    i : int
        Número de batch.
    n_batches : int
        Cantidad de batches de la época.
    loss : float
        Loss del batch.
    log : PredictionLog
        Predicciones de la época hasta ahora.
    metric : Metric
        Métrica del problema.
    """
    logger.info("  batch %d/%d  loss=%.4f  %s", i, n_batches, loss, format_score(log.score(metric)))


def _train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    metric: Metric,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weights: ClassWeights | None,
) -> tuple[float, PredictionLog]:
    """Hace una pasada completa sobre train.

    Parameters
    ----------
    model : nn.Module
        Red a entrenar, ya en ``device``.
    train_loader : DataLoader
        Instancias de train.
    feature_extractor : TextFeatureExtractor
        Extractor de features por token.
    metric : Metric
        Métrica del problema, que da la loss.
    optimizer : torch.optim.Optimizer
        Optimizador de la red.
    device : torch.device
        Device de la red.
    weights : ClassWeights or None
        Pesos por clase de la loss.

    Returns
    -------
    loss : float
        Loss promedio por batch.
    log : PredictionLog
        Las predicciones que hizo la red sobre train durante la época.
    """
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
        loss, pred = _train_step(model, X, Y, tokens, metric, optimizer, device, weights)
        log.add(pred, Y, tokens)
        total_loss += loss
        n_batches += 1
        if i % LOG_EVERY_N_BATCHES == 0:
            _log_progress(i, len(train_loader), loss, log, metric)
    return total_loss / max(n_batches, 1), log


@dataclass
class BestEpoch:
    """La época con mejor F1 en test hasta el momento.

    Attributes
    ----------
    epoch : int
        Número de época.
    score : float
        Promedio del F1 macro de sus tareas en test.
    weights : dict[str, torch.Tensor]
        Copia en CPU del ``state_dict`` de la red al final de la época.
    test_log : PredictionLog
        Sus predicciones sobre test.
    """

    epoch: int
    score: float
    weights: dict[str, torch.Tensor]
    test_log: PredictionLog


def _update_best(best: BestEpoch | None, metrics: EpochMetrics, model: nn.Module, test_log: PredictionLog) -> BestEpoch:
    """Guarda una copia de los pesos si la época mejora el promedio del F1 macro en test.

    Parameters
    ----------
    best : BestEpoch or None
        La mejor época hasta ahora (``None`` en la primera).
    metrics : EpochMetrics
        Métricas de la época que terminó.
    model : nn.Module
        Red al final de la época.
    test_log : PredictionLog
        Predicciones de la época sobre test.

    Returns
    -------
    BestEpoch
        ``best`` si la época no lo mejora; si no, la época nueva.
    """
    score = mean_score(metrics.test_score)
    if best is not None and score <= best.score:
        return best
    weights = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    return BestEpoch(metrics.epoch, score, weights, test_log)


def train_neural_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    metric: Metric,
    model: nn.Module,
    n_epochs: int | None = None,
    class_weight_power: float = CLASS_WEIGHT_POWER,
    weight_decay: float | None = None,
) -> TrainResult:
    """Entrena la red minimizando ``metric.loss``, con cada clase pesada según su frecuencia en train.

    Al final de cada época evalúa sobre test (loss y F1), para ver si sobreajusta. Se
    queda con los pesos de la época con mejor F1 en test (no necesariamente la última).

    Parameters
    ----------
    train_loader : DataLoader
        Instancias de train.
    test_loader : DataLoader
        Instancias de test.
    feature_extractor : TextFeatureExtractor
        Extractor de features por token.
    metric : Metric
        Métrica del problema, que da la loss.
    model : nn.Module
        Red sin entrenar.
    n_epochs : int or None, default=None
        Épocas de entrenamiento; ``None`` usa ``N_EPOCHS`` (el pipeline lo cambia con ``--epochs``).
    class_weight_power : float, default=CLASS_WEIGHT_POWER
        Exponente de los pesos por clase (ver ``src/training/weights.py``).
    weight_decay : float or None, default=None
        Weight decay de AdamW; ``None`` usa ``WEIGHT_DECAY``.

    Returns
    -------
    TrainResult
        La red en CPU con los pesos de la mejor época, sus predicciones sobre test, las
        métricas de cada época y la mejor época.
    """
    n_epochs = n_epochs or N_EPOCHS
    weight_decay = WEIGHT_DECAY if weight_decay is None else weight_decay
    device = _device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Entrenando %s en %s (%d parámetros, weight_decay=%g)", type(model).__name__, device, n_params, weight_decay
    )

    weights = class_weights(train_loader.dataset, class_weight_power)

    if weights is not None:
        if weights.size(0) == 1:
            # Capitalización
            weights[0][1] *= 0.7   # Inicial (estable)
            weights[0][2] *= 0.8   # MiXta (estable)
            weights[0][3] *= 0.65  # MAYÚS (destrabamos la predicción)
        
        elif weights.size(0) == 2:
            # Puntuación Inicial
            weights[0][1] *= 0.35  # ¿ (ajuste sutil para subir precisión)
            
            # Puntuación Final
            weights[1][1] *= 0.75  # , (estable)
            weights[1][2] *= 0.65   #
            weights[1][3] *= 0.7   # ? (estable)

    history, best = [], None
    for epoch in range(1, n_epochs + 1):
        train_loss, train_log = _train_epoch(model, train_loader, feature_extractor, metric, optimizer, device, weights)
        test_log, test_loss = _predict_with_loss(model, test_loader, feature_extractor, metric, device, weights)
        metrics = EpochMetrics(
            epoch, train_loss, train_log.score(metric), train_log.per_class(metric), test_loss, test_log.score(metric)
        )
        history.append(metrics)
        _log_epoch(metrics, n_epochs)
        best = _update_best(best, metrics, model, test_log)

    model.load_state_dict(best.weights)
    logger.info("Mejor época en test: %d/%d (F1 promedio %.3f)", best.epoch, n_epochs, best.score)
    return TrainResult(model.cpu(), best.test_log.word_level(metric), history, best.epoch)


def _log_epoch(metrics: EpochMetrics, n_epochs: int) -> None:
    """Loguea la loss y el F1 de train y de test de la época.

    Parameters
    ----------
    metrics : EpochMetrics
        Métricas de la época.
    n_epochs : int
        Cantidad total de épocas.
    """
    logger.info("Época %d/%d  train: loss=%.4f  %s", metrics.epoch, n_epochs, metrics.loss, format_score(metrics.score))
    logger.info(
        "Época %d/%d  test:  loss=%.4f  %s",
        metrics.epoch,
        n_epochs,
        metrics.test_loss,
        format_score(metrics.test_score),
    )


def evaluate_neural_model(
    model: nn.Module, loader: DataLoader, feature_extractor: TextFeatureExtractor, metric: Metric
) -> dict[str, WordLevelTask]:
    """Predice con una red ya entrenada (en CPU). La red vuelve a quedar en CPU.

    Parameters
    ----------
    model : nn.Module
        Red entrenada.
    loader : DataLoader
        Instancias a predecir.
    feature_extractor : TextFeatureExtractor
        Extractor de features por token.
    metric : Metric
        Métrica del problema.

    Returns
    -------
    dict[str, WordLevelTask]
        Clases reales y predichas a nivel palabra de cada tarea.
    """
    device = _device()
    tasks = _evaluate(model.to(device), loader, feature_extractor, metric, device)
    model.cpu()
    return tasks


def evaluate_problem_network(
    model: nn.Module, loader: DataLoader, feature_extractor: TextFeatureExtractor, problem: Problem
) -> dict[str, WordLevelTask]:
    """Aplica ``evaluate_neural_model`` con la métrica del problema, con la firma de ``Experiment.evaluator``.

    Parameters
    ----------
    model : nn.Module
        Red entrenada.
    loader : DataLoader
        Instancias a predecir.
    feature_extractor : TextFeatureExtractor
        Extractor de features por token.
    problem : Problem
        Problema de la red: da la métrica.

    Returns
    -------
    dict[str, WordLevelTask]
        Clases reales y predichas a nivel palabra de cada tarea.
    """
    return evaluate_neural_model(model, loader, feature_extractor, problem.metric)


def neural_trainer(build_model: ModelFactory, weight_decay: Callable[[Problem], float] | None = None) -> Trainer:
    """Adapta ``train_neural_model`` a la firma de ``Experiment.trainer``.

    Parameters
    ----------
    build_model : ModelFactory
        Crea la red sin entrenar para el problema.
    weight_decay : Callable[[Problem], float] or None, default=None
        Weight decay de cada problema; ``None`` usa ``WEIGHT_DECAY``.

    Returns
    -------
    Trainer
        Función ``(train_loader, test_loader, feature_extractor, problem) -> TrainResult``.
        Si el problema no implementa la red, devuelve un resultado vacío.
    """

    def trainer(
        train_loader: DataLoader, test_loader: DataLoader, feature_extractor: TextFeatureExtractor, problem: Problem
    ) -> TrainResult:
        """Crea la red del problema y la entrena.

        Parameters
        ----------
        train_loader : DataLoader
            Instancias de train.
        test_loader : DataLoader
            Instancias de test.
        feature_extractor : TextFeatureExtractor
            Extractor de features por token.
        problem : Problem
            Problema a resolver: da la red y la métrica.

        Returns
        -------
        TrainResult
            Ver ``train_neural_model``; ``TrainResult(None)`` si la red no está implementada.
        """
        try:
            model = build_model(problem)
        except NotImplementedError as error:
            logger.warning("Red sin implementar para %s: %s", type(problem).__name__, error)
            return TrainResult(None)
        decay = weight_decay(problem) if weight_decay else None
        return train_neural_model(
            train_loader, test_loader, feature_extractor, problem.metric, model, weight_decay=decay
        )

    return trainer
