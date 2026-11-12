import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import torch
from torch.utils.data import Dataset, DataLoader

from src.constants import MODELS_DIR, RUN_SEPARATOR
from src.data import build_loader, split_train_test
from src.features import (
    AverageEmbeddingsExtractor,
    ClassicFeatureExtractor,
    EmbeddingsExtractor,
    TextFeatureExtractor,
)
from src.problem import Problem
from src.training.classic import train_classic_model
from src.training.joint import train_joint_model
from src.training.neural import neural_trainer
from src.training.plots import save_class_metrics_plot, save_confusion_matrices, save_training_plot
from src.training.result import TrainResult, format_score

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Experiment:
    """Un modelo a entrenar junto con el extractor de features que recibe"""
    name: str
    trainer: Callable[[DataLoader, DataLoader, TextFeatureExtractor, Problem], TrainResult]
    feature_extractor: TextFeatureExtractor


def build_experiments() -> list[Experiment]:
    """Define las combinaciones de modelo y features que pide el enunciado"""
    embeddings = EmbeddingsExtractor()
    return [
        Experiment("classic_features", train_classic_model, ClassicFeatureExtractor()),
        Experiment("classic_embeddings", train_classic_model, AverageEmbeddingsExtractor()),
        Experiment("standard_rnn", neural_trainer(lambda problem: problem.standard_rnn_model()), embeddings),
        Experiment("bidirectional_rnn", neural_trainer(lambda problem: problem.bidirectional_rnn_model()), embeddings),
        Experiment("joint_rnn", train_joint_model, embeddings),
    ]


def _run_dir(problem_name: str) -> Path:
    """Carpeta del día de la corrida: models/<Problema>/<día>. Ej: 'models/PunctuationProblem/2026-11-12'"""
    model_dir = Path(MODELS_DIR) / problem_name / datetime.now().strftime("%Y-%m-%d")
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _file_stem(model_dir: Path, name: str, result: TrainResult, hour: str) -> Path:
    """<score>__<hora>__<modelo>, con el score como el promedio del F1 macro de sus tareas.
    Así, ordenados por nombre, los mejores modelos del día quedan al final. No usar
    with_suffix sobre el resultado: tomaría el punto del score como extensión"""
    mean_score = sum(result.score.values()) / len(result.score) if result.score else 0.0
    return model_dir / RUN_SEPARATOR.join((f"{mean_score:.3f}", hour, name))


def _with_ext(file_stem: Path, extension: str) -> Path:
    return file_stem.with_name(file_stem.name + extension)


def _save_experiment(name: str, result: TrainResult, file_stem: Path, problem_name: str) -> None:
    """Guarda el modelo entrenado y su gráfico. Las redes guardan sus pesos (.pkl) y los
    modelos clásicos el objeto completo (.joblib). El gráfico es la evolución por época si
    entrena por épocas, o la precision/recall por clase sobre test si no (modelos clásicos)"""
    if isinstance(result.model, torch.nn.Module):
        torch.save(result.model.state_dict(), _with_ext(file_stem, ".pkl"))
    elif result.model is not None:
        joblib.dump(result.model, _with_ext(file_stem, ".joblib"), compress=3)
    if result.history:
        save_training_plot(result.history, _with_ext(file_stem, ".png"), f"{problem_name} · {name} (train)")
    elif result.test_tasks:
        save_class_metrics_plot(result.per_class, _with_ext(file_stem, ".png"), f"{problem_name} · {name} (test)")


def train_models(
    development_dataset: Dataset,
    experiments: list[Experiment],
    problem: Problem
) -> dict[str, TrainResult]:
    """
    Acá se entrena y devuelve TODOS los modelos para el dataset dado.
    Inicialmente probamos los mismos modelos para ambos problemas.
    Al final guarda las matrices de confusión sobre test de todos los modelos.
    """
    train_dataset, test_dataset = split_train_test(development_dataset)
    logger.info("Split: train=%d, test=%d", len(train_dataset), len(test_dataset))
    train_loader = build_loader(train_dataset, shuffle=True)
    test_loader = build_loader(test_dataset, shuffle=False)

    problem_name = problem.__class__.__name__
    model_dir = _run_dir(problem_name)
    hour = datetime.now().strftime("%H-%M")
    results = {}
    for experiment in experiments:
        logger.info("Entrenando modelo %s", experiment.name)
        result = experiment.trainer(train_loader, test_loader, experiment.feature_extractor, problem)
        _save_experiment(experiment.name, result, _file_stem(model_dir, experiment.name, result, hour), problem_name)
        logger.info("Modelo %s terminado, test: %s", experiment.name, format_score(result.score))
        results[experiment.name] = result

    save_confusion_matrices(results, model_dir / f"{hour}{RUN_SEPARATOR}confusion_matrices.png", f"{problem_name} · matrices de confusión (test)")
    return results
