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
from src.metrics.base import WordLevelTask
from src.problem import Problem
from src.training.neural import Trainer
from src.training.result import TrainResult
from src.words import split_words, token_word_ids

logger = logging.getLogger(__name__)

LOG_EVERY_N_SENTENCES = 5000
N_FOLDS = 5


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


def classic_trainer(use_dictionary: bool) -> Trainer:
    """Adapta ``train_classic_model`` a la firma de ``Experiment.trainer``.

    Parameters
    ----------
    use_dictionary : bool
        Si el modelo usa el diccionario de capitalización.

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
            Ver ``train_classic_model``.
        """
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
