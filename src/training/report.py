"""Reporte de una corrida del pipeline: `models/<día>__<hora>/metrics.json`.

Guarda como datos (no solo como gráficos) todo lo necesario para comparar los modelos después,
por ejemplo desde notebooks/poster.ipynb:

    {
      "config": {...},                         # tamaños, ventana, hiperparámetros
      "data": {                                # instancias y palabras de cada split, por problema
        "CapitalizationProblem": {"train": {"instances": 9856, "words": 317204}, "test": {...}, "control": {...}}
      },
      "problems": {
        "CapitalizationProblem": {
          "bidirectional_rnn": {
            "test":    {"capitalizacion": {"f1_macro": 0.47, "confusion": [[...], ...]}},
            "control": {...},                  # mismo formato que test
            "best_epoch": 4,                   # None en los modelos clásicos
            "history": [{"epoch": 1, "loss": ..., "score": {...}, "test_score": {...}, ...}]
          }
        }
      }
    }

Por cada tarea se guarda la matriz de confusión a nivel palabra (fila = clase real, columna =
clase predicha): de ella salen la precision, el recall y el F1 de cada clase.
"""

import json
from pathlib import Path
from typing import Any

from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset

from src.data import split_dataset_dev_control, split_train_test
from src.data.samples import Sample
from src.metrics.base import WordLevelTask, macro_f1
from src.training.result import TrainResult

REPORT_FILE = "metrics.json"


def task_report(task: WordLevelTask) -> dict[str, Any]:
    """Calcula el F1 macro y la matriz de confusión de una tarea.

    Parameters
    ----------
    task : WordLevelTask
        Etiquetas reales, predicciones y cantidad de clases, a nivel palabra.

    Returns
    -------
    dict[str, Any]
        ``{"f1_macro": float, "confusion": list[list[int]]}``.
    """
    real, pred, n_classes = task
    return {
        "f1_macro": macro_f1(real, pred, n_classes),
        "confusion": confusion_matrix(real, pred, labels=range(n_classes)).tolist(),
    }


def tasks_report(tasks: dict[str, WordLevelTask]) -> dict[str, dict[str, Any]]:
    """Aplica ``task_report`` a cada tarea.

    Parameters
    ----------
    tasks : dict[str, WordLevelTask]
        Las tareas a nivel palabra.

    Returns
    -------
    dict[str, dict[str, Any]]
        El reporte de cada tarea.
    """
    return {name: task_report(task) for name, task in tasks.items()}


def experiment_report(result: TrainResult, control: TrainResult | None) -> dict[str, Any]:
    """Arma todo lo que se guarda de un modelo: test, control (si se evaluó) e historial por época.

    Parameters
    ----------
    result : TrainResult
        Resultado del entrenamiento, con las tareas de test.
    control : TrainResult or None
        Resultado de la evaluación sobre control, o ``None`` si no se evaluó.

    Returns
    -------
    dict[str, Any]
        ``{"test", "control", "best_epoch", "history"}``.
    """
    return {
        "test": tasks_report(result.test_tasks),
        "control": tasks_report(control.test_tasks) if control else None,
        "best_epoch": result.best_epoch,
        "history": [epoch._asdict() for epoch in result.history],
    }


def data_report(samples: list[Sample]) -> dict[str, dict[str, int]]:
    """Cuenta instancias y palabras de train, test y control.

    Usa los mismos splits que el pipeline, que dependen solo de la cantidad de
    instancias y la semilla.

    Parameters
    ----------
    samples : list[Sample]
        Todas las instancias del problema.

    Returns
    -------
    dict[str, dict[str, int]]
        Para cada split, ``{"instances": int, "words": int}``.
    """
    development, control = split_dataset_dev_control(samples)
    train, test = split_train_test(development)
    return {name: _split_size(split) for name, split in (("train", train), ("test", test), ("control", control))}


def _split_size(split: Dataset) -> dict[str, int]:
    """Cuenta las instancias y las palabras de un split.

    Parameters
    ----------
    split : Dataset
        Split de la lista de instancias.

    Returns
    -------
    dict[str, int]
        ``{"instances": int, "words": int}``.
    """
    return {"instances": len(split), "words": sum(len(split[i]) for i in range(len(split)))}


def save_report(run_dir: Path, report: dict[str, Any]) -> Path:
    """Escribe el reporte en ``run_dir/metrics.json`` (se reescribe entero en cada llamada).

    Parameters
    ----------
    run_dir : Path
        Carpeta de la corrida.
    report : dict[str, Any]
        Reporte completo.

    Returns
    -------
    Path
        Path del archivo escrito.
    """
    path = run_dir / REPORT_FILE
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return path
