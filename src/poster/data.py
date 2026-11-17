"""Datos para los gráficos del póster: el reporte (metrics.json) de una corrida del pipeline.

Las corridas nuevas lo guardan solas (ver src/training/report.py). Para una carpeta vieja que
no lo tiene, `load_run` vuelve a evaluar sus modelos sobre test y control, con los mismos
splits del pipeline, y lo guarda en la carpeta para no repetirlo. Para eso hay que pasarle
los tamaños y la ventana con que se entrenó (los splits dependen de ellos). Una corrida vieja
no tiene historial por época, y no se pueden evaluar sus `classic_features` (de antes de que
los modelos clásicos pasaran a nivel palabra, con otros atributos).
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from src.constants import CAPITALIZATION_DATASET_SIZE, MODELS_DIR, PUNCTUATION_DATASET_SIZE, RUN_SEPARATOR
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
from src.inference import available_models, load_model, model_name
from src.metrics import Metric
from src.metrics.base import WordLevelTask
from src.problem import CapitalizationProblem, Problem, PunctuationProblem
from src.training import Experiment, build_experiments
from src.training.classic import ClassicModel
from src.training.report import REPORT_FILE, data_report, save_report, tasks_report
from src.words import concat_word_ids, token_word_ids

logger = logging.getLogger(__name__)

Report = dict[str, Any]
Evaluate = Callable[[DataLoader], dict[str, WordLevelTask]]


def latest_run(models_dir: str | Path = MODELS_DIR) -> Path:
    """Busca la corrida más reciente con modelos de algún problema.

    Las carpetas se llaman ``<día>__<hora>``, así que la última en orden alfabético es
    la más nueva.

    Parameters
    ----------
    models_dir : str or Path, default=MODELS_DIR
        Carpeta con las corridas.

    Returns
    -------
    Path
        Carpeta de la corrida.

    Raises
    ------
    FileNotFoundError
        Si no hay ninguna corrida con modelos.
    """
    runs = sorted(
        run
        for run in Path(models_dir).glob(f"*{RUN_SEPARATOR}*")
        if any(available_models(run / problem) for problem in ("CapitalizationProblem", "PunctuationProblem"))
    )
    if not runs:
        raise FileNotFoundError(f"No hay corridas con modelos en {models_dir}/: correr antes src.pipeline")
    return runs[-1]


def load_run(
    run_dir: str | Path,
    capitalization_size: int = CAPITALIZATION_DATASET_SIZE,
    punctuation_size: int = PUNCTUATION_DATASET_SIZE,
    window: WindowSpec | None = WindowSpec(),
) -> Report:
    """Carga el reporte de la corrida, armándolo si no existe.

    Si la carpeta no tiene ``metrics.json``, lo arma reevaluando sus modelos con esos
    tamaños y ventana (los de la corrida). En los dos casos lo guarda.

    Parameters
    ----------
    run_dir : str or Path
        Carpeta de la corrida.
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
    run_dir = Path(run_dir)
    path = run_dir / REPORT_FILE
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
    else:
        logger.warning("%s no tiene %s: reevaluando sus modelos (tarda unos minutos)", run_dir, REPORT_FILE)
        report = reevaluate_run(run_dir, capitalization_size, punctuation_size, window)
    if "data" not in report:  # corridas de antes de que el pipeline guardara los tamaños
        report["data"] = data_sizes(report["config"])
    logger.info("Reporte guardado en %s", save_report(run_dir, report))
    return report


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


def reevaluate_run(run_dir: Path, capitalization_size: int, punctuation_size: int, window: WindowSpec | None) -> Report:
    """Evalúa sobre test y control todos los modelos de la carpeta que se puedan cargar.

    Parameters
    ----------
    run_dir : Path
        Carpeta de la corrida.
    capitalization_size : int
        Instancias por fuente con que se entrenó capitalización.
    punctuation_size : int
        Instancias por fuente con que se entrenó puntuación.
    window : WindowSpec or None
        Ventana con que se cortaron las instancias.

    Returns
    -------
    Report
        Reporte con ``"config"`` y ``"problems"`` (sin historial por época).
    """
    experiments = {experiment.name: experiment for experiment in build_experiments()}
    punctuation, capitalization = preprocess_dataset(punctuation_size, capitalization_size, window)
    config = {
        "capitalization_size": capitalization_size,
        "punctuation_size": punctuation_size,
        "window": vars(window) if window else None,
        "reevaluated": True,
    }
    report: Report = {"config": config, "problems": {}}
    for dataset, problem in [(capitalization, CapitalizationProblem()), (punctuation, PunctuationProblem())]:
        development, control = split_dataset_dev_control(dataset)
        _, test = split_train_test(development)
        loaders = {"test": build_loader(test, shuffle=False), "control": build_loader(control, shuffle=False)}
        problem_name = type(problem).__name__
        models = {}
        for path in available_models(run_dir / problem_name):
            if (model_report := _reevaluate_model(path, problem, experiments, loaders)) is not None:
                models[model_name(path)] = model_report
        report["problems"][problem_name] = models
    return report


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
