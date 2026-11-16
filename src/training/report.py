"""Reporte de una corrida del pipeline: `models/<día>__<hora>/metrics.json`.

Guarda como datos (no solo como gráficos) todo lo necesario para comparar los modelos después,
por ejemplo desde notebooks/poster.ipynb:

    {
      "config": {...},                         # tamaños, ventana, hiperparámetros
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

from src.metrics.base import WordLevelTask, macro_f1
from src.training.result import TrainResult

REPORT_FILE = "metrics.json"


def task_report(task: WordLevelTask) -> dict[str, Any]:
    """F1 macro y matriz de confusión de una tarea"""
    real, pred, n_classes = task
    return {
        "f1_macro": macro_f1(real, pred, n_classes),
        "confusion": confusion_matrix(real, pred, labels=range(n_classes)).tolist(),
    }


def tasks_report(tasks: dict[str, WordLevelTask]) -> dict[str, dict[str, Any]]:
    return {name: task_report(task) for name, task in tasks.items()}


def experiment_report(result: TrainResult, control: TrainResult | None) -> dict[str, Any]:
    """Todo lo que se guarda de un modelo: test, control (si se evaluó) e historial por época"""
    return {
        "test": tasks_report(result.test_tasks),
        "control": tasks_report(control.test_tasks) if control else None,
        "best_epoch": result.best_epoch,
        "history": [epoch._asdict() for epoch in result.history],
    }


def save_report(run_dir: Path, report: dict[str, Any]) -> Path:
    """Escribe el reporte en run_dir/metrics.json (se reescribe entero en cada llamada)"""
    path = run_dir / REPORT_FILE
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return path
