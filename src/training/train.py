import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import torch
from torch.utils.data import Dataset, DataLoader

# Algunos imports solo los usan los experimentos comentados en build_experiments
from src.constants import RUNS_DIR, RUN_SEPARATOR
from src.data import build_loader, split_train_test
from src.features import (
    AverageEmbeddingsExtractor,
    ClassicFeatureExtractor,
    EmbeddingsExtractor,
    TextFeatureExtractor,
)
from src.problem import Problem
from src.training.classic import classic_trainer, evaluate_classic_model, train_classic_model
from src.problem.joint import (
    create_joint_model,
    create_conv_bilstm_model,
    create_projected_bigru_model,
    create_residual_stacked_bilstm_model,
)
from src.training.joint import evaluate_joint_model, joint_trainer
from src.training.neural import Evaluator, evaluate_problem_network, neural_trainer
from src.training.plots import save_class_metrics_plot, save_confusion_matrices, save_training_plot
from src.training.result import TrainResult, format_score, mean_score

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Experiment:
    """Un modelo a entrenar, el extractor de features que recibe y cómo predecir con él una vez entrenado.

    Attributes
    ----------
    name : str
        Nombre del experimento, con el que se guarda el modelo.
    trainer : Callable[[DataLoader, DataLoader, TextFeatureExtractor, Problem], TrainResult]
        Entrena el modelo sobre train y lo evalúa sobre test.
    evaluator : Evaluator
        Predice con el modelo ya entrenado (para evaluarlo sobre control).
    feature_extractor : TextFeatureExtractor
        Extractor de features que recibe el modelo.
    """

    name: str
    trainer: Callable[[DataLoader, DataLoader, TextFeatureExtractor, Problem], TrainResult]
    evaluator: Evaluator
    feature_extractor: TextFeatureExtractor


def build_experiments() -> list[Experiment]:
    """Define las combinaciones de modelo y features que pide el enunciado.

    Returns
    -------
    list[Experiment]
        Los experimentos activos, en el orden en que se entrenan.
    """
    embeddings = EmbeddingsExtractor()
    return [
        # Experiment("classic_features", train_classic_model, evaluate_classic_model, ClassicFeatureExtractor()),
        # Experiment(
        #     "classic_features_dictionary",
        #     classic_trainer(use_dictionary=True),
        #     evaluate_classic_model,
        #     ClassicFeatureExtractor(),
        # ),
        # Experiment("classic_embeddings", train_classic_model, evaluate_classic_model, AverageEmbeddingsExtractor()),
        # Random forest con atributos + diccionario que crece de a poco: una iteración por época
        # (--epochs), hasta los 100 árboles del modelo clásico. Así tiene curvas de aprendizaje
        Experiment(
            "classic_features_dictionary",
            classic_trainer(use_dictionary=True, by_iterations=True),
            evaluate_classic_model,
            ClassicFeatureExtractor(),
        ),
        Experiment(
            "standard_rnn",
            neural_trainer(lambda problem: problem.standard_rnn_model()),
            evaluate_problem_network,
            embeddings,
        ),
        Experiment(
            "bidirectional_rnn",
            neural_trainer(
                lambda problem: problem.bidirectional_rnn_model(
                    problem.bidirectional_rnn_dropout, problem.bidirectional_rnn_head_dim
                ),
                weight_decay=lambda problem: problem.bidirectional_rnn_weight_decay,
            ),
            evaluate_problem_network,
            embeddings,
        ),
        # Experiment(
        #     "joint_rnn",
        #     joint_trainer(lambda: create_joint_model(bidirectional=False)),
        #     evaluate_joint_model,
        #     embeddings,
        # ),
        # Experiment(
        #     "bidirectional_joint_rnn",
        #     joint_trainer(lambda: create_joint_model(bidirectional=True)),
        #     evaluate_joint_model,
        #     embeddings,
        # ),
        # Experiment(
        #     "joint_lstm",
        #     joint_trainer(lambda: create_joint_model(bidirectional=False, lstm=True)),
        #     evaluate_joint_model,
        #     embeddings,
        # ),
        # Experiment(
        #     "bidirectional_joint_lstm",
        #     joint_trainer(lambda: create_joint_model(bidirectional=True, lstm=True)),
        #     evaluate_joint_model,
        #     embeddings,
        # ),
        # # Variantes de la red conjunta, ver src/problem/joint/
        # Experiment(
        #     "joint_projected_bigru", joint_trainer(create_projected_bigru_model), evaluate_joint_model, embeddings
        # ),
        # Experiment(
        #     "joint_residual_stacked_bilstm",
        #     joint_trainer(create_residual_stacked_bilstm_model),
        #     evaluate_joint_model,
        #     embeddings,
        # ),
        Experiment(
            "joint_conv_bilstm", 
            joint_trainer(create_conv_bilstm_model), 
            evaluate_joint_model, 
            embeddings
        ),
    ]


def new_run() -> str:
    """Nombre de una corrida nueva del pipeline: ``<día>__<hora>``.

    Se guarda en la metadata de cada modelo que entrena la corrida.

    Returns
    -------
    str
        Ej: ``'2026-11-12__19-30-05'``.
    """
    return datetime.now().strftime(f"%Y-%m-%d{RUN_SEPARATOR}%H-%M-%S")


def model_stem(name: str, result: TrainResult) -> str:
    """Arma el nombre base del modelo: ``<score>__<modelo>``.

    El score es el promedio del F1 macro de sus tareas sobre test. Así, ordenados por
    nombre, los mejores modelos de cada problema quedan al final.

    Parameters
    ----------
    name : str
        Nombre del experimento.
    result : TrainResult
        Resultado del entrenamiento.

    Returns
    -------
    str
        Ej: ``'0.812__standard_rnn'``.
    """
    return RUN_SEPARATOR.join((f"{mean_score(result.score):.3f}", name))


def model_directory(problem_name: str, name: str, result: TrainResult, run: str) -> Path:
    """Crea la carpeta del modelo: ``models/runs/<Problema>/<score>__<modelo>/``.

    Si ya hay un modelo con el mismo score y nombre, se le agrega la corrida al final.

    Parameters
    ----------
    problem_name : str
        Nombre del problema.
    name : str
        Nombre del experimento.
    result : TrainResult
        Resultado del entrenamiento.
    run : str
        Corrida (ver ``new_run``).

    Returns
    -------
    Path
        Ej: ``models/runs/PunctuationProblem/0.812__standard_rnn``.
    """
    folder = Path(RUNS_DIR) / problem_name / model_stem(name, result)
    if folder.exists():
        folder = folder.with_name(RUN_SEPARATOR.join((folder.name, run)))
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _with_ext(file_stem: Path, extension: str) -> Path:
    """Agrega una extensión al nombre base sin interpretar el punto del score.

    Parameters
    ----------
    file_stem : Path
        Path sin extensión, ej: ``.../0.812__standard_rnn``.
    extension : str
        Extensión con el punto, ej: ``".pkl"``.

    Returns
    -------
    Path
        El path con la extensión agregada.
    """
    return file_stem.with_name(file_stem.name + extension)


def save_experiment(name: str, result: TrainResult, model_dir: Path, problem_name: str) -> None:
    """Guarda en ``model_dir`` el modelo entrenado y sus gráficos sobre test.

    Las redes guardan sus pesos (``<score>__<modelo>.pkl``) y los modelos clásicos el objeto
    completo (``.joblib``). El gráfico es la evolución por época si entrena por épocas
    (``train_test.png``), o la precision/recall por clase sobre test si no (``test.png``).
    Además van sus matrices de confusión sobre test (``test_confusion.png``).

    Parameters
    ----------
    name : str
        Nombre del experimento.
    result : TrainResult
        Resultado del entrenamiento.
    model_dir : Path
        Carpeta del modelo (ver ``model_directory``).
    problem_name : str
        Nombre del problema, para el título de los gráficos.
    """
    file_stem = model_dir / model_stem(name, result)
    if isinstance(result.model, torch.nn.Module):
        torch.save(result.model.state_dict(), _with_ext(file_stem, ".pkl"))
    elif result.model is not None:
        joblib.dump(result.model, _with_ext(file_stem, ".joblib"), compress=3)
    title = f"{problem_name} · {name}"
    if result.history:
        save_training_plot(result.history, model_dir / "train_test.png", f"{title} (train y test)", result.best_epoch)
    elif result.test_tasks:
        save_class_metrics_plot(result.per_class, model_dir / "test.png", f"{title} (test)")
    if result.test_tasks:
        save_confusion_matrices({name: result}, model_dir / "test_confusion.png", f"{title} (test)")


def train_models(
    development_dataset: Dataset, experiments: list[Experiment], problem: Problem, run: str
) -> tuple[dict[str, TrainResult], dict[str, Path]]:
    """Entrena todos los experimentos sobre el dataset de desarrollo y guarda cada uno en su carpeta.

    Divide el desarrollo en train y test, entrena cada experimento y lo guarda en
    ``models/runs/<Problema>/<score>__<modelo>/``.

    Parameters
    ----------
    development_dataset : Dataset
        Dataset de desarrollo del problema.
    experiments : list[Experiment]
        Experimentos a entrenar.
    problem : Problem
        Problema a resolver.
    run : str
        Corrida (ver ``new_run``).

    Returns
    -------
    tuple[dict[str, TrainResult], dict[str, Path]]
        Resultado y carpeta de cada experimento, por nombre.
    """
    train_dataset, test_dataset = split_train_test(development_dataset)
    logger.info("Split: train=%d, test=%d", len(train_dataset), len(test_dataset))
    train_loader = build_loader(train_dataset, shuffle=True)
    test_loader = build_loader(test_dataset, shuffle=False)

    problem_name = problem.__class__.__name__
    results, folders = {}, {}
    for experiment in experiments:
        logger.info("Entrenando modelo %s", experiment.name)
        result = experiment.trainer(train_loader, test_loader, experiment.feature_extractor, problem)
        folders[experiment.name] = model_directory(problem_name, experiment.name, result, run)
        save_experiment(experiment.name, result, folders[experiment.name], problem_name)
        logger.info("Modelo %s terminado, test: %s", experiment.name, format_score(result.score))
        results[experiment.name] = result
    return results, folders


def evaluate_on_control(
    control_dataset: Dataset, experiment: Experiment, result: TrainResult, problem: Problem, model_dir: Path
) -> TrainResult:
    """Evalúa sobre control el modelo ya entrenado del experimento.

    Guarda en ``model_dir`` su precision/recall por clase (``control.png``) y sus matrices
    de confusión (``control_confusion.png``).

    Parameters
    ----------
    control_dataset : Dataset
        Dataset de control del problema.
    experiment : Experiment
        Experimento del modelo.
    result : TrainResult
        Resultado del entrenamiento, con el modelo.
    problem : Problem
        Problema del modelo.
    model_dir : Path
        Carpeta del modelo.

    Returns
    -------
    TrainResult
        Resultado cuyas tareas son las de control.
    """
    loader = build_loader(control_dataset, shuffle=False)
    control = TrainResult(
        result.model, experiment.evaluator(result.model, loader, experiment.feature_extractor, problem)
    )

    title = f"{problem.__class__.__name__} · {experiment.name} (control)"
    save_class_metrics_plot(control.per_class, model_dir / "control.png", title)
    save_confusion_matrices({experiment.name: control}, model_dir / "control_confusion.png", title)
    return control
