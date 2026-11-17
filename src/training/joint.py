"""Experimentos conjuntos (`joint_rnn`, `bidirectional_joint_lstm`, las variantes de
src/problem/joint/, etc.): una red que aprende capitalización y puntuación a la vez.

Las demás redes se entrenan solo con las etiquetas del problema. Esta necesita las 3
etiquetas de cada token, así que se arma un `JointDataset` con las mismas oraciones y se le
aplican exactamente los mismos splits (desarrollo/control y train/test) que al dataset del
problema. Después se evalúan solo las cabezas del problema, para compararla con el resto.
"""

from collections.abc import Callable

from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.data import JointDataset, build_loader
from src.features import TextFeatureExtractor
from src.metrics import JointMetric
from src.metrics.base import WordLevelTask
from src.problem import Problem
from src.training.neural import Trainer, evaluate_neural_model, train_neural_model
from src.training.result import TrainResult


def _root_dataset(dataset: Dataset) -> Dataset:
    """Busca el dataset original debajo de los ``Subset`` que generan los splits.

    Parameters
    ----------
    dataset : Dataset
        Un dataset o un ``Subset`` (anidado o no) de uno.

    Returns
    -------
    Dataset
        El dataset original.
    """
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def _same_split(dataset: Dataset, new_root: Dataset) -> Dataset:
    """Repite sobre ``new_root`` los mismos ``Subset`` (los mismos índices) que tiene ``dataset``.

    Parameters
    ----------
    dataset : Dataset
        Split de referencia.
    new_root : Dataset
        Dataset con las mismas instancias, en el mismo orden, que el original de ``dataset``.

    Returns
    -------
    Dataset
        ``new_root`` con los mismos splits que ``dataset``.
    """
    if isinstance(dataset, Subset):
        return Subset(_same_split(dataset.dataset, new_root), dataset.indices)
    return new_root


def _joint_loader(loader: DataLoader, joint_dataset: Dataset, shuffle: bool) -> DataLoader:
    """Crea un loader con las mismas instancias que ``loader``, pero con las 3 etiquetas.

    Parameters
    ----------
    loader : DataLoader
        Loader del problema.
    joint_dataset : Dataset
        ``JointDataset`` con las instancias del dataset original de ``loader``.
    shuffle : bool
        Si se mezclan las instancias en cada época.

    Returns
    -------
    DataLoader
        Loader del ``JointDataset`` con el mismo split.
    """
    return build_loader(_same_split(loader.dataset, joint_dataset), shuffle)


def _joint_dataset(loader: DataLoader) -> Dataset:
    """Arma un ``JointDataset`` con las instancias del dataset original de ``loader``.

    Parameters
    ----------
    loader : DataLoader
        Loader del problema.

    Returns
    -------
    Dataset
        Las mismas instancias, en el mismo orden, con las 3 etiquetas.
    """
    return JointDataset(_root_dataset(loader.dataset).samples)


def train_joint_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    problem: Problem,
    build_model: Callable[[], nn.Module],
) -> TrainResult:
    """Entrena una red conjunta con las 3 tareas y la evalúa solo en las tareas del problema.

    Parameters
    ----------
    train_loader : DataLoader
        Instancias de train del problema.
    test_loader : DataLoader
        Instancias de test del problema.
    feature_extractor : TextFeatureExtractor
        Extractor de features por token.
    problem : Problem
        Problema en el que se evalúa la red.
    build_model : Callable[[], nn.Module]
        Crea la red conjunta sin entrenar.

    Returns
    -------
    TrainResult
        La red entrenada; las tareas de test son solo las del problema y el historial
        tiene las 3.
    """
    joint_dataset = _joint_dataset(train_loader)
    result = train_neural_model(
        _joint_loader(train_loader, joint_dataset, shuffle=True),
        _joint_loader(test_loader, joint_dataset, shuffle=False),
        feature_extractor,
        JointMetric(),
        build_model(),
    )
    return result.only_tasks(problem.metric.tasks)


def joint_trainer(build_model: Callable[[], nn.Module]) -> Trainer:
    """Adapta ``train_joint_model`` a la firma de ``Experiment.trainer``.

    Parameters
    ----------
    build_model : Callable[[], nn.Module]
        Crea la red sin entrenar, con salida ``(batch, n_tokens, 3, 4)`` como ``JointRNN``.

    Returns
    -------
    Trainer
        Función ``(train_loader, test_loader, feature_extractor, problem) -> TrainResult``.
    """

    def trainer(
        train_loader: DataLoader, test_loader: DataLoader, feature_extractor: TextFeatureExtractor, problem: Problem
    ) -> TrainResult:
        """Entrena la red conjunta que crea ``build_model``.

        Parameters
        ----------
        train_loader : DataLoader
            Instancias de train del problema.
        test_loader : DataLoader
            Instancias de test del problema.
        feature_extractor : TextFeatureExtractor
            Extractor de features por token.
        problem : Problem
            Problema en el que se evalúa la red.

        Returns
        -------
        TrainResult
            Ver ``train_joint_model``.
        """
        return train_joint_model(train_loader, test_loader, feature_extractor, problem, build_model)

    return trainer


def evaluate_joint_model(
    model: nn.Module, loader: DataLoader, feature_extractor: TextFeatureExtractor, problem: Problem
) -> dict[str, WordLevelTask]:
    """Predice con una red conjunta ya entrenada y devuelve solo las tareas del problema.

    Parameters
    ----------
    model : nn.Module
        Red conjunta entrenada.
    loader : DataLoader
        Instancias a predecir.
    feature_extractor : TextFeatureExtractor
        Extractor de features por token.
    problem : Problem
        Problema cuyas tareas se devuelven.

    Returns
    -------
    dict[str, WordLevelTask]
        Clases reales y predichas a nivel palabra de las tareas del problema.
    """
    joint_loader = _joint_loader(loader, _joint_dataset(loader), shuffle=False)
    tasks = evaluate_neural_model(model, joint_loader, feature_extractor, JointMetric())
    return {task: tasks[task] for task in problem.metric.tasks}
