"""Datos para los gráficos del póster: el reporte de una corrida del pipeline.

El reporte se arma con la metadata.json de cada modelo de la corrida (ver src/training/report.py).
Las corridas nuevas la guardan con sus métricas. Para un modelo viejo sin métricas, `load_run`
lo vuelve a evaluar sobre test y control, con los mismos splits del pipeline, y las guarda en su
metadata para no repetirlo. Para eso hay que pasarle los tamaños y la ventana con que se entrenó
(los splits dependen de ellos). Un modelo reevaluado no tiene historial por época, y no se pueden
evaluar los `classic_features` viejos (de antes de que los modelos clásicos pasaran a nivel
palabra, con otros atributos).
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.constants import CAPITALIZATION_DATASET_SIZE, PUNCTUATION_DATASET_SIZE, RUNS_DIR
from src.data import (
    build_loader,
    load_documents,
    load_samples,
    preprocess_dataset,
    split_dataset_dev_control,
    split_train_test,
)
from src.data.samples import WindowSpec
from src.features import AverageEmbeddingsExtractor, TextFeatureExtractor
from src.inference import load_model, model_name
from src.metrics import Metric
from src.metrics.base import WordLevelTask
from src.problem import CapitalizationProblem, Problem, PunctuationProblem
from src.training import Experiment, build_experiments
from src.training.classic import ClassicModel
from src.training.report import (
    data_report,
    list_runs,
    load_metadata,
    load_report,
    run_models,
    save_metadata,
    tasks_report,
)
from src.words import concat_word_ids, token_word_ids

logger = logging.getLogger(__name__)

Report = dict[str, Any]
Evaluate = Callable[[DataLoader], dict[str, WordLevelTask]]


def latest_run(runs_dir: str | Path = RUNS_DIR) -> str:
    """Busca la corrida ``<día>__<hora>`` más reciente con algún modelo.

    Parameters
    ----------
    runs_dir : str or Path, default=RUNS_DIR
        Carpeta con los modelos.

    Returns
    -------
    str
        La corrida, ej: ``'2026-11-12__19-30-05'``.

    Raises
    ------
    FileNotFoundError
        Si no hay ninguna corrida con modelos.
    """
    runs = [run for run in list_runs(runs_dir) if run[:1].isdigit()]
    if not runs:
        raise FileNotFoundError(f"No hay corridas con modelos en {runs_dir}/: correr antes src.pipeline")
    return runs[-1]


def load_run(
    run: str,
    capitalization_size: int = CAPITALIZATION_DATASET_SIZE,
    punctuation_size: int = PUNCTUATION_DATASET_SIZE,
    window: WindowSpec | None = WindowSpec(),
) -> Report:
    """Carga el reporte de la corrida, completando lo que le falte a la metadata de sus modelos.

    Los modelos sin métricas se reevalúan con esos tamaños y ventana (los de la corrida), y los
    que no tienen los tamaños de los splits los recalculan. En los dos casos se guarda en su metadata.

    Parameters
    ----------
    run : str
        Corrida, ej: ``'2026-11-16__13-37-08'``.
    capitalization_size : int, default=CAPITALIZATION_DATASET_SIZE
        Instancias por fuente con que se entrenó capitalización.
    punctuation_size : int, default=PUNCTUATION_DATASET_SIZE
        Instancias por fuente con que se entrenó puntuación.
    window : WindowSpec or None, default=WindowSpec()
        Ventana con que se cortaron las instancias.

    Returns
    -------
    Report
        El reporte, con el formato de ``src/training/report.py``.
    """
    folders = run_models(run)
    if not folders:
        raise FileNotFoundError(f"No hay modelos de la corrida {run} en {RUNS_DIR}/")
    missing = [folder for folder in folders if load_metadata(folder).get("test") is None]
    if missing:
        logger.warning("%d modelos de %s no tienen métricas: reevaluando (tarda unos minutos)", len(missing), run)
        reevaluate_models(missing, capitalization_size, punctuation_size, window)
    _fill_data_sizes(folders)
    return load_report(folders)


def _fill_data_sizes(folders: list[Path]) -> None:
    """Agrega los tamaños de los splits a la metadata de los modelos que no los tienen.

    Parameters
    ----------
    folders : list[Path]
        Carpetas de los modelos de una corrida.
    """
    without = [folder for folder in folders if not load_metadata(folder).get("data")]
    if not without or not load_metadata(without[0]).get("config"):  # sin config no se pueden rearmar
        return
    sizes = data_sizes(load_metadata(without[0])["config"])  # corridas de antes de que se guardaran
    for folder in without:
        metadata = load_metadata(folder)
        save_metadata(folder, {**metadata, "data": sizes[metadata["problem"]]})


def data_sizes(config: dict[str, Any]) -> dict[str, dict[str, dict[str, int]]]:
    """Cuenta instancias y palabras de train, test y control de cada problema.

    Rearma las instancias con los tamaños y la ventana de la corrida (los splits son
    los mismos que en el pipeline).

    Parameters
    ----------
    config : dict[str, Any]
        Sección ``"config"`` del reporte.

    Returns
    -------
    dict[str, dict[str, dict[str, int]]]
        Por problema, el ``data_report`` de sus instancias.
    """
    window = WindowSpec(**config["window"]) if config.get("window") else None
    documents = load_documents()
    sizes = {"CapitalizationProblem": config["capitalization_size"], "PunctuationProblem": config["punctuation_size"]}
    return {problem: data_report(load_samples(documents, size, window)) for problem, size in sizes.items()}


def reevaluate_models(
    folders: list[Path], capitalization_size: int, punctuation_size: int, window: WindowSpec | None
) -> None:
    """Evalúa sobre test y control los modelos de esas carpetas y guarda las métricas en su metadata.

    Parameters
    ----------
    folders : list[Path]
        Carpetas de los modelos.
    capitalization_size : int
        Instancias por fuente con que se entrenó capitalización.
    punctuation_size : int
        Instancias por fuente con que se entrenó puntuación.
    window : WindowSpec or None
        Ventana con que se cortaron las instancias.
    """
    experiments = {experiment.name: experiment for experiment in build_experiments()}
    punctuation, capitalization = preprocess_dataset(punctuation_size, capitalization_size, window)
    config = {
        "capitalization_size": capitalization_size,
        "punctuation_size": punctuation_size,
        "window": vars(window) if window else None,
        "reevaluated": True,
    }
    for dataset, problem in [(capitalization, CapitalizationProblem()), (punctuation, PunctuationProblem())]:
        own = [folder for folder in folders if load_metadata(folder)["problem"] == type(problem).__name__]
        if own:
            loaders = _test_control_loaders(dataset)
            for folder in own:
                _reevaluate_folder(folder, problem, experiments, loaders, config)


def _test_control_loaders(dataset: Dataset) -> dict[str, DataLoader]:
    """Arma los loaders de test y control con los mismos splits del pipeline.

    Parameters
    ----------
    dataset : Dataset
        Todas las instancias del problema.

    Returns
    -------
    dict[str, DataLoader]
        ``{"test": ..., "control": ...}``.
    """
    development, control = split_dataset_dev_control(dataset)
    _, test = split_train_test(development)
    return {"test": build_loader(test, shuffle=False), "control": build_loader(control, shuffle=False)}


def _reevaluate_folder(
    folder: Path, problem: Problem, experiments: dict[str, Experiment], loaders: dict[str, DataLoader], config: Report
) -> None:
    """Reevalúa el modelo de una carpeta y guarda sus métricas en su metadata.

    Parameters
    ----------
    folder : Path
        Carpeta del modelo.
    problem : Problem
        Problema del modelo.
    experiments : dict[str, Experiment]
        Experimentos por nombre, para saber cómo evaluarlo.
    loaders : dict[str, DataLoader]
        Loaders de test y control.
    config : Report
        Config a guardar si la metadata no tiene una.
    """
    path = next(path for path in folder.iterdir() if path.suffix in (".pkl", ".joblib"))
    if (results := _reevaluate_model(path, problem, experiments, loaders)) is not None:
        metadata = load_metadata(folder)
        save_metadata(folder, {**metadata, **results, "config": metadata.get("config") or config})


def _reevaluate_model(
    path: Path, problem: Problem, experiments: dict[str, Experiment], loaders: dict[str, DataLoader]
) -> Report | None:
    """Arma el reporte de un modelo guardado, sin historial.

    Parameters
    ----------
    path : Path
        Archivo del modelo.
    problem : Problem
        Problema del modelo.
    experiments : dict[str, Experiment]
        Experimentos por nombre, para saber cómo evaluarlo.
    loaders : dict[str, DataLoader]
        Loaders de test y control.

    Returns
    -------
    Report or None
        El reporte de test y control, o ``None`` si no se puede evaluar.
    """
    name = model_name(path)
    model, _ = load_model(path, problem, name)
    evaluate = _evaluator(model, experiments[name], problem)
    if evaluate is None:
        logger.warning("Se saltea %s: es un modelo clásico de antes de pasar a nivel palabra", path.name)
        return None
    logger.info("Evaluando %s", path.name)
    splits = {split: tasks_report(evaluate(loader)) for split, loader in loaders.items()}
    return {**splits, "best_epoch": None, "history": []}


def _evaluator(model: Any, experiment: Experiment, problem: Problem) -> Evaluate | None:
    """Decide cómo evaluar el modelo.

    Los clásicos viejos (el estimador solo, entrenado por token) se pueden evaluar si
    usaban el promedio de embeddings, que da las mismas features que antes.

    Parameters
    ----------
    model : Any
        Modelo cargado.
    experiment : Experiment
        Experimento del modelo.
    problem : Problem
        Problema del modelo.

    Returns
    -------
    Evaluate or None
        Función ``loader -> tareas a nivel palabra``, o ``None`` si no se puede evaluar.
    """
    extractor = experiment.feature_extractor
    if isinstance(model, (nn.Module, ClassicModel)):
        return lambda loader: experiment.evaluator(model, loader, extractor, problem)
    if isinstance(extractor, AverageEmbeddingsExtractor):
        return lambda loader: _evaluate_token_level(model, loader, extractor, problem.metric)
    return None


def _evaluate_token_level(
    estimator: Any, loader: DataLoader, extractor: TextFeatureExtractor, metric: Metric
) -> dict[str, WordLevelTask]:
    """Evalúa un clásico viejo: una fila por token, y la métrica pasa a nivel palabra.

    Parameters
    ----------
    estimator : Any
        Estimador de sklearn entrenado por token.
    loader : DataLoader
        Instancias a predecir.
    extractor : TextFeatureExtractor
        Extractor con el que se entrenó.
    metric : Metric
        Métrica del problema.

    Returns
    -------
    dict[str, WordLevelTask]
        Clases reales y predichas a nivel palabra de cada tarea.
    """
    X, Y, word_ids = [], [], []
    for i in range(len(loader.dataset)):
        tokens, labels = loader.dataset[i]
        if tokens:
            X.append(extractor.extract(tokens).numpy())
            Y.append(labels.numpy())
            word_ids.append(token_word_ids(tokens))
    return metric.word_level(estimator.predict(np.concatenate(X)), np.concatenate(Y), concat_word_ids(word_ids))
