from dataclasses import dataclass, field
from typing import Any, NamedTuple

from src.metrics.base import PerClass, WordLevelTask, per_class_from_tasks, scores_from_tasks


class EpochMetrics(NamedTuple):
    """Métricas de una época.

    Attributes
    ----------
    epoch : int
        Número de época.
    loss : float
        Loss promedio sobre train.
    score : dict[str, float]
        F1 macro de cada tarea sobre train.
    per_class : PerClass
        ``(precision, recall)`` de cada clase sobre train.
    test_loss : float
        Loss promedio sobre test, con la red al final de la época y los mismos pesos
        por clase que train.
    test_score : dict[str, float]
        F1 macro de cada tarea sobre test.
    """

    epoch: int
    loss: float
    score: dict[str, float]
    per_class: PerClass
    test_loss: float
    test_score: dict[str, float]


@dataclass
class TrainResult:
    """Lo que devuelve cada trainer.

    Attributes
    ----------
    model : Any
        El modelo entrenado (``None`` si la red no está implementada).
    test_tasks : dict[str, WordLevelTask]
        Sus predicciones sobre test, a nivel palabra.
    history : list[EpochMetrics]
        Métricas de cada época; vacío en los modelos que no entrenan por épocas.
    best_epoch : int or None
        Época con la que se quedó la red; ``None`` en los modelos clásicos.
    """

    model: Any
    test_tasks: dict[str, WordLevelTask] = field(default_factory=dict, repr=False)
    history: list[EpochMetrics] = field(default_factory=list, repr=False)
    best_epoch: int | None = None

    @property
    def score(self) -> dict[str, float]:
        """F1 macro sobre test de cada tarea.

        Returns
        -------
        dict[str, float]
            F1 macro de cada tarea.
        """
        return scores_from_tasks(self.test_tasks)

    @property
    def per_class(self) -> PerClass:
        """Precision y recall sobre test de cada clase de cada tarea.

        Returns
        -------
        PerClass
            Por tarea, ``(precision, recall)`` de cada clase.
        """
        return per_class_from_tasks(self.test_tasks)

    def only_tasks(self, tasks: tuple[str, ...]) -> "TrainResult":
        """Devuelve el mismo resultado, evaluado solo en esas tareas (el historial queda completo).

        Parameters
        ----------
        tasks : tuple[str, ...]
            Tareas que se conservan.

        Returns
        -------
        TrainResult
            Un resultado nuevo con el mismo modelo e historial.
        """
        kept = {task: self.test_tasks[task] for task in tasks}
        return TrainResult(self.model, kept, self.history, self.best_epoch)


def mean_score(score: dict[str, float]) -> float:
    """Resume el F1 macro de cada tarea en un número: el promedio.

    Parameters
    ----------
    score : dict[str, float]
        F1 macro de cada tarea.

    Returns
    -------
    float
        El promedio, o 0 si no hay tareas.
    """
    return sum(score.values()) / len(score) if score else 0.0


def format_score(score: dict[str, float]) -> str:
    """Formatea el F1 de cada tarea para los logs.

    Parameters
    ----------
    score : dict[str, float]
        F1 macro de cada tarea.

    Returns
    -------
    str
        Ej: ``'punt_inicial=0.912 punt_final=0.655'``.
    """
    return " ".join(f"{task}={value:.3f}" for task, value in score.items())
