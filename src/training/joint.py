"""Experimento `joint_rnn`: una RNN que aprende capitalización y puntuación a la vez.

Las demás redes se entrenan solo con las etiquetas del problema. Esta necesita las 3
etiquetas de cada token, así que se arma un `JointDataset` con las mismas oraciones y se le
aplican exactamente los mismos splits (desarrollo/control y train/test) que al dataset del
problema. Después se evalúan solo las cabezas del problema, para compararla con el resto.
"""
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.data import build_loader
from src.features import TextFeatureExtractor
from src.joint_rnn import JointDataset, create_joint_rnn_model
from src.metrics import JointMetric
from src.metrics.base import WordLevelTask
from src.problem import Problem
from src.training.neural import evaluate_neural_model, train_neural_model
from src.training.result import TrainResult


def _root_dataset(dataset: Dataset) -> Dataset:
    """El dataset original debajo de los Subset que generan los splits"""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def _same_split(dataset: Dataset, new_root: Dataset) -> Dataset:
    """Repite sobre `new_root` los mismos Subset (los mismos índices) que tiene `dataset`"""
    if isinstance(dataset, Subset):
        return Subset(_same_split(dataset.dataset, new_root), dataset.indices)
    return new_root


def _joint_loader(loader: DataLoader, joint_dataset: Dataset, shuffle: bool) -> DataLoader:
    """Un loader igual a `loader`, con las mismas oraciones, pero con las 3 etiquetas"""
    return build_loader(_same_split(loader.dataset, joint_dataset), shuffle)


def _joint_dataset(loader: DataLoader) -> Dataset:
    """Las mismas oraciones del dataset original de `loader`, en el mismo orden, con las 3 etiquetas"""
    root = _root_dataset(loader.dataset)
    return JointDataset(root.path, len(root))


def train_joint_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    problem: Problem
) -> TrainResult:
    """Entrena JointRNN con las 3 tareas y la evalúa solo en las tareas del problema"""
    joint_dataset = _joint_dataset(train_loader)
    result = train_neural_model(
        _joint_loader(train_loader, joint_dataset, shuffle=True),
        _joint_loader(test_loader, joint_dataset, shuffle=False),
        feature_extractor,
        JointMetric(),
        create_joint_rnn_model()
    )
    return result.only_tasks(problem.metric.tasks)


def evaluate_joint_model(
    model: nn.Module,
    loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    problem: Problem
) -> dict[str, WordLevelTask]:
    """Predice con JointRNN ya entrenada sobre las oraciones de `loader` y devuelve solo
    las tareas del problema"""
    joint_loader = _joint_loader(loader, _joint_dataset(loader), shuffle=False)
    tasks = evaluate_neural_model(model, joint_loader, feature_extractor, JointMetric())
    return {task: tasks[task] for task in problem.metric.tasks}
