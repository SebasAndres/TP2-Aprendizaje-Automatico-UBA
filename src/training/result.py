from dataclasses import dataclass, field
from typing import Any, NamedTuple

from src.metrics.base import PerClass, WordLevelTask, per_class_from_tasks, scores_from_tasks


class EpochMetrics(NamedTuple):
    """Métricas de una época. Sobre train: loss promedio, score del problema y
    (precision, recall) por clase. Sobre test, con la red al final de la época: loss
    promedio (con los mismos pesos por clase que train) y score del problema"""

    epoch: int
    loss: float
    score: dict[str, float]
    per_class: PerClass
    test_loss: float
    test_score: dict[str, float]


@dataclass
class TrainResult:
    """Lo que devuelve cada trainer: el modelo, sus predicciones sobre test a nivel palabra
    y, si entrena por épocas, las métricas de cada época y la época con la que se quedó"""

    model: Any
    test_tasks: dict[str, WordLevelTask] = field(default_factory=dict, repr=False)
    history: list[EpochMetrics] = field(default_factory=list, repr=False)
    best_epoch: int | None = None

    @property
    def score(self) -> dict[str, float]:
        """F1 macro sobre test de cada tarea"""
        return scores_from_tasks(self.test_tasks)

    @property
    def per_class(self) -> PerClass:
        """(precision, recall) sobre test de cada clase de cada tarea"""
        return per_class_from_tasks(self.test_tasks)

    def only_tasks(self, tasks: tuple[str, ...]) -> "TrainResult":
        """El mismo resultado, pero evaluado solo en esas tareas (el historial queda completo)"""
        kept = {task: self.test_tasks[task] for task in tasks}
        return TrainResult(self.model, kept, self.history, self.best_epoch)


def mean_score(score: dict[str, float]) -> float:
    """El F1 macro de cada tarea resumido en un número: el promedio (0 si no hay tareas)"""
    return sum(score.values()) / len(score) if score else 0.0


def format_score(score: dict[str, float]) -> str:
    """Ej: 'punt_inicial=0.912 punt_final=0.655'"""
    return " ".join(f"{task}={value:.3f}" for task, value in score.items())
