"""Modelos clásicos (no secuenciales), entrenados a nivel palabra.

Cada fila es una palabra: sus features (`extract_words` del extractor) y sus etiquetas
(`word_labels` de la métrica: la capitalización de la palabra, la puntuación inicial de su
primer token y la final del último). Así una palabra de varios tokens es una sola fila, como
en la evaluación, y no hay filas con las mismas features y distinta etiqueta (con el
promedio de embeddings, todos los tokens de `cu ##ánd ##o?` tenían la misma X, pero solo el
último tenía el "?"). Al predecir, la clase de cada palabra se copia a todos sus tokens.

Opcionalmente se agregan las features del diccionario de capitalización (ver
src/features/dictionary.py), armado solo con train. Para las filas de train no se usa el
diccionario completo: una palabra vista una sola vez traería su propia etiqueta como feature y
el modelo aprendería a confiar ciegamente en el diccionario, cosa que en test no se cumple. Por
eso train se divide en N_FOLDS partes y cada oración usa el diccionario armado con las otras
partes. Test, control e inferencia usan el diccionario de todo train.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.data import dataset_samples
from src.features import TextFeatureExtractor
from src.features.dictionary import CapitalizationDictionary
from src.metrics import Metric
from src.metrics.base import WordLevelTask, per_class_from_tasks, scores_from_tasks
from src.problem import Problem
from src.training import neural
from src.training.neural import Trainer
from src.training.result import EpochMetrics, TrainResult
from src.words import split_words, token_word_ids

logger = logging.getLogger(__name__)

LOG_EVERY_N_SENTENCES = 5000
N_FOLDS = 5
PROBA_EPS = 1e-7  # piso de las probabilidades en la log loss, para que no dé infinito


@dataclass
class ClassicModel:
    """Lo que se guarda de un modelo clásico.

    Attributes
    ----------
    estimator : Any
        Estimador de sklearn, entrenado a nivel palabra.
    dictionary : CapitalizationDictionary or None, default=None
        Diccionario de capitalización armado con train, si el modelo lo usa.
    """

    estimator: Any
    dictionary: CapitalizationDictionary | None = None

    def features(self, extractor: TextFeatureExtractor, tokens: list[str]) -> np.ndarray:
        """Calcula las features de cada palabra: las del extractor, más las del diccionario si hay.

        Parameters
        ----------
        extractor : TextFeatureExtractor
            Extractor con el que se entrenó el modelo.
        tokens : list[str]
            Tokens BERT de una instancia.

        Returns
        -------
        np.ndarray of shape (n_palabras, n_features)
            Las features de cada palabra.
        """
        X = extractor.extract_words(tokens).numpy()
        if self.dictionary is None:
            return X
        return np.hstack([X, self.dictionary.features(split_words(tokens))])

    def predict_words(self, extractor: TextFeatureExtractor, tokens: list[str]) -> np.ndarray:
        """Predice la clase de cada palabra.

        Parameters
        ----------
        extractor : TextFeatureExtractor
            Extractor con el que se entrenó el modelo.
        tokens : list[str]
            Tokens BERT de una instancia.

        Returns
        -------
        np.ndarray of shape (n_palabras,) or (n_palabras, n_tareas)
            La clase predicha de cada palabra.
        """
        return self.estimator.predict(self.features(extractor, tokens))


def _dataset_to_numpy(
    dataset: Dataset, feature_extractor: TextFeatureExtractor, metric: Metric, model_for: Callable[[int], ClassicModel]
) -> tuple[np.ndarray, np.ndarray]:
    """Arma las matrices de features y etiquetas a nivel palabra de todo el dataset.

    Omite las instancias sin tokens.

    Parameters
    ----------
    dataset : Dataset
        Dataset de pares ``(tokens, etiquetas)``.
    feature_extractor : TextFeatureExtractor
        Extractor de features por palabra.
    metric : Metric
        Métrica del problema, que pasa las etiquetas a nivel palabra.
    model_for : Callable[[int], ClassicModel]
        De qué modelo toma sus features la instancia ``i`` (ver ``_train_features_source``).

    Returns
    -------
    X : np.ndarray of shape (n_palabras, n_features)
        Features de cada palabra.
    Y : np.ndarray of shape (n_palabras,) or (n_palabras, n_tareas)
        Etiquetas de cada palabra.
    """
    rows_X, rows_Y = [], []
    for i in range(len(dataset)):
        tokens, labels = dataset[i]
        if not tokens:
            continue
        rows_X.append(model_for(i).features(feature_extractor, tokens))
        rows_Y.append(metric.word_labels(labels.numpy(), token_word_ids(tokens)))
        if (i + 1) % LOG_EVERY_N_SENTENCES == 0:
            logger.info("Extrayendo features: oración %d/%d", i + 1, len(dataset))
    X, Y = np.concatenate(rows_X), np.concatenate(rows_Y)
    logger.info("Features: %d/%d oraciones, X=%s, Y=%s", len(rows_X), len(dataset), X.shape, Y.shape)
    return X, Y


def _train_features_source(train_set: Dataset, model: ClassicModel) -> Callable[[int], ClassicModel]:
    """Decide de qué modelo toma sus features cada instancia de train.

    Sin diccionario, del modelo mismo. Con diccionario, la instancia ``i`` usa uno
    armado sin su parte (``i % N_FOLDS``) de train.

    Parameters
    ----------
    train_set : Dataset
        Instancias de train.
    model : ClassicModel
        Modelo a entrenar.

    Returns
    -------
    Callable[[int], ClassicModel]
        El modelo (con su diccionario) que corresponde a la instancia ``i``.
    """
    if model.dictionary is None:
        return lambda i: model
    samples = dataset_samples(train_set)
    folds = [
        ClassicModel(
            model.estimator,
            CapitalizationDictionary([sample for j, sample in enumerate(samples) if j % N_FOLDS != fold]),
        )
        for fold in range(N_FOLDS)
    ]
    return lambda i: folds[i % N_FOLDS]


def train_classic_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    problem: Problem,
    use_dictionary: bool = False,
) -> TrainResult:
    """Ajusta el modelo clásico del problema sobre train y lo evalúa sobre test.

    No entrena por épocas, así que no tiene historial.

    Parameters
    ----------
    train_loader : DataLoader
        Instancias de train.
    test_loader : DataLoader
        Instancias de test.
    feature_extractor : TextFeatureExtractor
        Extractor con el que se calculan las features por palabra.
    problem : Problem
        Problema a resolver: da el modelo clásico y la métrica.
    use_dictionary : bool, default=False
        Si agrega las features del diccionario de capitalización.

    Returns
    -------
    TrainResult
        El ``ClassicModel`` ajustado y sus predicciones sobre test a nivel palabra.
    """
    train_set = train_loader.dataset
    model = ClassicModel(problem.classic_model())
    if use_dictionary:
        logger.info("Armando el diccionario de capitalización con train")
        model.dictionary = CapitalizationDictionary(dataset_samples(train_set))
    logger.info("Extrayendo features de train")
    X_train, Y_train = _dataset_to_numpy(
        train_set, feature_extractor, problem.metric, _train_features_source(train_set, model)
    )
    logger.info("Ajustando %s", type(model.estimator).__name__)
    model.estimator.fit(X_train, Y_train)

    logger.info("Prediciendo sobre test")
    return TrainResult(model, evaluate_classic_model(model, test_loader, feature_extractor, problem))


def _trees_per_iteration(n_estimators: int, n_iterations: int) -> list[int]:
    """Cuántos árboles tiene el bosque al final de cada iteración.

    Parameters
    ----------
    n_estimators : int
        Árboles del bosque completo.
    n_iterations : int
        Cantidad de iteraciones.

    Returns
    -------
    list[int]
        Ej: ``(100, 20)`` -> ``[5, 10, ..., 100]``. La última es siempre ``n_estimators``.
    """
    return [max(1, round(n_estimators * (i + 1) / n_iterations)) for i in range(n_iterations)]


class _ForestVotes:
    """Suma de las probabilidades de los árboles de un bosque sobre un conjunto fijo de filas.

    Así, al agregar árboles con ``warm_start``, solo se predice con los nuevos. El promedio de
    las probabilidades de los árboles es lo que usa ``RandomForestClassifier.predict``.

    Parameters
    ----------
    X : np.ndarray of shape (n_filas, n_features)
        Filas a predecir.
    """

    def __init__(self, X: np.ndarray) -> None:
        self.X = X.astype(np.float32)  # el tipo que usan los árboles por dentro
        self.sums: list[np.ndarray] | None = None
        self.n_trees = 0

    def add(self, trees: list[Any]) -> None:
        """Suma las probabilidades de ``trees``.

        Parameters
        ----------
        trees : list
            Árboles nuevos del bosque.
        """
        for tree in trees:
            proba = tree.predict_proba(self.X, check_input=False)
            proba = proba if isinstance(proba, list) else [proba]  # una salida por tarea
            self.sums = proba if self.sums is None else [s + p for s, p in zip(self.sums, proba)]
            self.n_trees += 1

    def predict(self, forest: Any) -> tuple[np.ndarray, list[np.ndarray]]:
        """Predice como ``forest.predict`` con los árboles sumados hasta ahora.

        Parameters
        ----------
        forest : RandomForestClassifier
            Bosque, para pasar de índice de clase a clase.

        Returns
        -------
        pred : np.ndarray of shape (n_filas,) or (n_filas, n_tareas)
            Clase predicha de cada fila.
        proba : list[np.ndarray]
            Probabilidad de cada clase, una matriz por tarea.
        """
        proba = [s / self.n_trees for s in self.sums]
        classes = forest.classes_ if forest.n_outputs_ > 1 else [forest.classes_]
        pred = np.stack([c[p.argmax(axis=1)] for c, p in zip(classes, proba)], axis=1)
        return (pred[:, 0] if forest.n_outputs_ == 1 else pred), proba


def _log_loss(forest: Any, proba: list[np.ndarray], Y: np.ndarray) -> float:
    """Log loss promedio por palabra, sumada sobre las tareas (como la loss de las redes conjuntas).

    Parameters
    ----------
    forest : RandomForestClassifier
        Bosque, para ubicar la columna de cada clase.
    proba : list[np.ndarray]
        Probabilidad de cada clase, una matriz por tarea.
    Y : np.ndarray of shape (n_filas,) or (n_filas, n_tareas)
        Clases reales.

    Returns
    -------
    float
        La log loss, sin pesos por clase.
    """
    classes = forest.classes_ if forest.n_outputs_ > 1 else [forest.classes_]
    Y = Y.reshape(len(Y), -1)
    total = 0.0
    for task, (c, p) in enumerate(zip(classes, proba)):
        columns = np.searchsorted(c, Y[:, task])
        total += -np.log(np.clip(p[np.arange(len(Y)), columns], PROBA_EPS, 1.0)).mean()
    return float(total)


def train_classic_model_by_iterations(
    train_loader: DataLoader,
    test_loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    problem: Problem,
    use_dictionary: bool = False,
    n_iterations: int | None = None,
) -> TrainResult:
    """Como ``train_classic_model``, pero hace crecer el bosque de a poco y guarda un historial.

    El random forest no entrena por épocas: en cada iteración se le agregan árboles
    (``warm_start``) hasta llegar a los ``n_estimators`` de ``problem.classic_model()``, y se
    mide el F1 en train y en test, igual que las redes en cada época. Con la misma semilla, el
    bosque final es el mismo que el de ``train_classic_model``.

    El F1 de train se calcula con las mismas features con que se ajustó (en train, el diccionario
    de cada oración no incluye su parte, ver ``_train_features_source``). Los árboles crecen hasta
    el fondo, así que en train queda muy cerca de 1. La loss es la log loss sin pesos por clase:
    no se compara con la de las redes.

    Parameters
    ----------
    train_loader : DataLoader
        Instancias de train.
    test_loader : DataLoader
        Instancias de test.
    feature_extractor : TextFeatureExtractor
        Extractor con el que se calculan las features por palabra.
    problem : Problem
        Problema a resolver: da el modelo clásico y la métrica.
    use_dictionary : bool, default=False
        Si agrega las features del diccionario de capitalización.
    n_iterations : int or None, default=None
        Iteraciones; ``None`` usa ``neural.N_EPOCHS`` (el pipeline lo cambia con ``--epochs``).

    Returns
    -------
    TrainResult
        El ``ClassicModel`` con todos los árboles, sus predicciones sobre test y el historial
        por iteración. ``best_epoch`` es la última: el modelo guardado es el bosque completo.
    """
    n_iterations = n_iterations or neural.N_EPOCHS
    train_set = train_loader.dataset
    model = ClassicModel(problem.classic_model())
    forest = model.estimator
    if use_dictionary:
        logger.info("Armando el diccionario de capitalización con train")
        model.dictionary = CapitalizationDictionary(dataset_samples(train_set))
    logger.info("Extrayendo features de train")
    X_train, Y_train = _dataset_to_numpy(
        train_set, feature_extractor, problem.metric, _train_features_source(train_set, model)
    )
    logger.info("Extrayendo features de test")
    X_test, Y_test = _dataset_to_numpy(test_loader.dataset, feature_extractor, problem.metric, lambda i: model)

    train_votes, test_votes = _ForestVotes(X_train), _ForestVotes(X_test)
    history: list[EpochMetrics] = []
    forest.set_params(warm_start=True)
    for iteration, n_trees in enumerate(_trees_per_iteration(forest.n_estimators, n_iterations), start=1):
        before = len(getattr(forest, "estimators_", []))
        forest.set_params(n_estimators=n_trees)
        forest.fit(X_train, Y_train)
        new_trees = forest.estimators_[before:]
        train_votes.add(new_trees)
        test_votes.add(new_trees)

        train_pred, train_proba = train_votes.predict(forest)
        test_pred, test_proba = test_votes.predict(forest)
        train_tasks = problem.metric.word_level(train_pred, Y_train, np.arange(len(Y_train)))
        test_tasks = problem.metric.word_level(test_pred, Y_test, np.arange(len(Y_test)))
        metrics = EpochMetrics(
            epoch=iteration,
            loss=_log_loss(forest, train_proba, Y_train),
            score=scores_from_tasks(train_tasks),
            per_class=per_class_from_tasks(train_tasks),
            test_loss=_log_loss(forest, test_proba, Y_test),
            test_score=scores_from_tasks(test_tasks),
        )
        history.append(metrics)
        logger.info(
            "Iteración %d/%d (%d árboles): train %s · test %s",
            iteration,
            n_iterations,
            n_trees,
            " ".join(f"{t}={v:.3f}" for t, v in metrics.score.items()),
            " ".join(f"{t}={v:.3f}" for t, v in metrics.test_score.items()),
        )
    forest.set_params(warm_start=False)
    return TrainResult(model, test_tasks, history, best_epoch=n_iterations)


def classic_trainer(use_dictionary: bool, by_iterations: bool = False) -> Trainer:
    """Adapta ``train_classic_model`` a la firma de ``Experiment.trainer``.

    Parameters
    ----------
    use_dictionary : bool
        Si el modelo usa el diccionario de capitalización.
    by_iterations : bool, default=False
        Si usa ``train_classic_model_by_iterations`` (bosque que crece por iteraciones, con
        historial para las curvas de aprendizaje).

    Returns
    -------
    Trainer
        Función ``(train_loader, test_loader, feature_extractor, problem) -> TrainResult``.
    """

    def trainer(
        train_loader: DataLoader, test_loader: DataLoader, feature_extractor: TextFeatureExtractor, problem: Problem
    ) -> TrainResult:
        """Entrena el modelo clásico con ``use_dictionary`` fijo.

        Parameters
        ----------
        train_loader : DataLoader
            Instancias de train.
        test_loader : DataLoader
            Instancias de test.
        feature_extractor : TextFeatureExtractor
            Extractor con el que se calculan las features por palabra.
        problem : Problem
            Problema a resolver: da el modelo clásico y la métrica.

        Returns
        -------
        TrainResult
            Ver ``train_classic_model`` y ``train_classic_model_by_iterations``.
        """
        if by_iterations:
            return train_classic_model_by_iterations(
                train_loader, test_loader, feature_extractor, problem, use_dictionary
            )
        return train_classic_model(train_loader, test_loader, feature_extractor, problem, use_dictionary)

    return trainer


def evaluate_classic_model(
    model: ClassicModel, loader: DataLoader, feature_extractor: TextFeatureExtractor, problem: Problem
) -> dict[str, WordLevelTask]:
    """Predice con un modelo clásico ya ajustado.

    Parameters
    ----------
    model : ClassicModel
        Modelo ajustado.
    loader : DataLoader
        Instancias a predecir.
    feature_extractor : TextFeatureExtractor
        Extractor con el que se entrenó el modelo.
    problem : Problem
        Problema del modelo: da la métrica.

    Returns
    -------
    dict[str, WordLevelTask]
        Clases reales y predichas a nivel palabra de cada tarea.
    """
    X, Y = _dataset_to_numpy(loader.dataset, feature_extractor, problem.metric, lambda i: model)
    # Cada fila ya es una palabra: cada "token" es su propia palabra
    return problem.metric.word_level(model.estimator.predict(X), Y, np.arange(len(Y)))
