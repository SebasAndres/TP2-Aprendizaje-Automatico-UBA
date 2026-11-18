"""Metadata de cada modelo y reporte de una corrida del pipeline.

Cada modelo guarda su ``metadata.json`` en ``models/runs/<Problema>/<score>__<modelo>/``:

    {"run": "2026-11-18__04-38-20", "problem": "PunctuationProblem", "model": "joint_conv_bilstm",
     "config": {...}, "data": {...},          # config de la corrida y tamaños de los splits del problema
     "test": {...}, "control": {...}, "best_epoch": 18, "history": [...]}

``run_report`` / ``load_report`` juntan esas metadatas en el reporte de una corrida (o de varias),
con el formato que usan los gráficos, por ejemplo desde notebooks/poster.ipynb:

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

from src.constants import RUNS_DIR
from src.data import split_dataset_dev_control, split_train_test
from src.data.samples import Sample
from src.metrics.base import WordLevelTask, macro_f1
from src.training.result import TrainResult

REPORT_FILE = "metrics.json"
METADATA_FILE = "metadata.json"
RESULT_KEYS = ("test", "control", "best_epoch", "history")

Report = dict[str, Any]


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


def model_metadata(run: str, problem: str, model: str, report: Report) -> Report:
    """Arma la metadata de un modelo a partir del reporte de su corrida.

    Parameters
    ----------
    run : str
        Corrida, ej: ``'2026-11-18__04-38-20'``.
    problem : str
        Nombre del problema.
    model : str
        Nombre del experimento.
    report : Report
        Reporte de la corrida, con ``"config"``, ``"data"`` y ``"problems"``.

    Returns
    -------
    Report
        ``{"run", "problem", "model", "config", "data", "test", "control", "best_epoch", "history"}``.
    """
    results = report["problems"][problem][model]
    return {
        "run": run,
        "problem": problem,
        "model": model,
        "config": report["config"],
        "data": report["data"].get(problem),
        **{key: results.get(key) for key in RESULT_KEYS},
    }


def save_metadata(model_dir: Path, metadata: Report) -> Path:
    """Escribe ``model_dir/metadata.json``.

    Parameters
    ----------
    model_dir : Path
        Carpeta del modelo.
    metadata : Report
        Metadata del modelo (ver ``model_metadata``).

    Returns
    -------
    Path
        Path del archivo escrito.
    """
    path = model_dir / METADATA_FILE
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_metadata(model_dir: Path) -> Report:
    """Lee ``model_dir/metadata.json``.

    Parameters
    ----------
    model_dir : Path
        Carpeta del modelo.

    Returns
    -------
    Report
        Metadata del modelo.
    """
    return json.loads((Path(model_dir) / METADATA_FILE).read_text(encoding="utf-8"))


def model_dirs(runs_dir: str | Path = RUNS_DIR) -> list[Path]:
    """Lista las carpetas de modelos: ``runs_dir/<Problema>/<score>__<modelo>/``.

    Parameters
    ----------
    runs_dir : str or Path, default=RUNS_DIR
        Carpeta con los modelos.

    Returns
    -------
    list[Path]
        Las carpetas que tienen ``metadata.json``, ordenadas.
    """
    return sorted(path.parent for path in Path(runs_dir).glob(f"*/*/{METADATA_FILE}"))


def list_runs(runs_dir: str | Path = RUNS_DIR) -> list[str]:
    """Lista las corridas que tienen algún modelo guardado.

    Parameters
    ----------
    runs_dir : str or Path, default=RUNS_DIR
        Carpeta con los modelos.

    Returns
    -------
    list[str]
        Las corridas, ordenadas (las ``<día>__<hora>`` quedan de la más vieja a la más nueva).
    """
    return sorted({load_metadata(folder)["run"] for folder in model_dirs(runs_dir)})


def run_models(run: str, runs_dir: str | Path = RUNS_DIR) -> list[Path]:
    """Busca las carpetas de los modelos de una corrida.

    Parameters
    ----------
    run : str
        Corrida, ej: ``'2026-11-18__04-38-20'``.
    runs_dir : str or Path, default=RUNS_DIR
        Carpeta con los modelos.

    Returns
    -------
    list[Path]
        Las carpetas de sus modelos.
    """
    return [folder for folder in model_dirs(runs_dir) if load_metadata(folder)["run"] == run]


def load_report(folders: list[Path]) -> Report:
    """Junta la metadata de varios modelos en un reporte con el formato de una corrida.

    La config es la del primer modelo. Si los modelos vienen de más de una corrida,
    ``config["merged_from"]`` dice de cuáles.

    Parameters
    ----------
    folders : list[Path]
        Carpetas de los modelos.

    Returns
    -------
    Report
        ``{"config", "data", "problems": {problema: {modelo: {"test", "control", ...}}}}``.
    """
    metadatas = [load_metadata(folder) for folder in folders]
    report: Report = {"config": dict(metadatas[0]["config"] or {}), "data": {}, "problems": {}}
    runs = list(dict.fromkeys(metadata["run"] for metadata in metadatas))
    if len(runs) > 1:
        report["config"]["merged_from"] = runs
    for metadata in metadatas:
        if metadata.get("data"):
            report["data"][metadata["problem"]] = metadata["data"]
        results = {key: metadata.get(key) for key in RESULT_KEYS}
        report["problems"].setdefault(metadata["problem"], {})[metadata["model"]] = results
    return report


def run_report(run: str, runs_dir: str | Path = RUNS_DIR) -> Report:
    """Arma el reporte de una corrida con la metadata de sus modelos.

    Parameters
    ----------
    run : str
        Corrida, ej: ``'2026-11-18__04-38-20'``.
    runs_dir : str or Path, default=RUNS_DIR
        Carpeta con los modelos.

    Returns
    -------
    Report
        El reporte (ver ``load_report``).

    Raises
    ------
    FileNotFoundError
        Si la corrida no tiene modelos.
    """
    folders = run_models(run, runs_dir)
    if not folders:
        raise FileNotFoundError(f"No hay modelos de la corrida {run} en {runs_dir}/")
    return load_report(folders)
