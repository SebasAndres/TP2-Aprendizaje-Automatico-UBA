"""Gráficos comparativos de los modelos del candidato final (ver candidato_final_resultados.ipynb).

El candidato final se entrenó con una corrida del pipeline por modelo. `merge_runs` junta la
metadata de los modelos de esas corridas (models/runs/<Problema>/<score>__<modelo>/) en un solo reporte con el formato de
`src/training/report.py`, y el resto de las funciones dibuja a partir de ese reporte. Usa la
paleta y el estilo de `src/poster/plots.py` para que los gráficos combinen con los del póster.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.poster import plots as P
from src.training.plots import SERIES_COLORS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY
from src.training.report import load_report, run_models

Report = dict[str, Any]
Scores = dict[tuple[str, str], float]  # (modelo, tarea) -> F1 macro

MODELS = ["joint_conv_bilstm", "bidirectional_rnn"]  # de mejor a peor en control


def merge_runs(runs: list[str]) -> Report:
    """Junta los modelos de varias corridas del pipeline en un solo reporte.

    Parameters
    ----------
    runs : list[str]
        Corridas, ej: ``["2026-11-18__04-44-00", "2026-11-18__04-38-20"]``. Tienen que
        compartir datos y configuración.

    Returns
    -------
    Report
        Config y datos de la primera corrida, y los modelos de todas.
    """
    report = load_report([folder for run in runs for folder in run_models(run)])
    report["config"]["merged_from"] = list(runs)
    return report


def control_scores(report: Report, split: str = "control") -> Scores:
    """F1 macro de cada modelo de ``MODELS`` en cada tarea.

    Parameters
    ----------
    report : Report
        Reporte de una corrida.
    split : str, default="control"
        ``"test"`` o ``"control"``.

    Returns
    -------
    Scores
        F1 macro por (modelo, tarea).
    """
    return {
        (model, task): v["f1_macro"]
        for models in report["problems"].values()
        for model, r in models.items()
        if model in MODELS and r and r.get(split)
        for task, v in r[split].items()
    }


def best_previous(reports: list[Report], split: str = "control") -> Scores:
    """Mejor resultado previo de cada modelo y problema, entre varias corridas.

    Para cada modelo y problema se queda con la corrida de mejor F1 promedio entre las tareas
    del problema (no con el máximo de cada tarea por separado, que mezclaría corridas).

    Parameters
    ----------
    reports : list[Report]
        Reportes de las corridas previas (ver ``merge_runs`` y ``src.training.report.run_report``).
    split : str, default="control"
        ``"test"`` o ``"control"``.

    Returns
    -------
    Scores
        F1 macro por (modelo, tarea).
    """
    best: Scores = {}
    best_mean: dict[tuple[str, str], float] = {}
    for report in reports:
        for problem, models in report["problems"].items():
            for model in MODELS:
                tasks = (models.get(model) or {}).get(split)
                mean = np.mean([v["f1_macro"] for v in tasks.values()]) if tasks else -1.0
                if mean > best_mean.get((model, problem), -1.0):
                    best_mean[(model, problem)] = mean
                    best.update({(model, task): v["f1_macro"] for task, v in tasks.items()})
    return best


def precision_recall(confusion: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Precisión y recall de cada clase a partir de la matriz de confusión (filas = real).

    Parameters
    ----------
    confusion : list[list[int]]
        Matriz de confusión con conteos.

    Returns
    -------
    precision : np.ndarray
        Precisión de cada clase (0 si nunca se predijo).
    recall : np.ndarray
        Recall de cada clase (0 si no aparece).
    """
    counts = np.array(confusion, dtype=float)
    hits = np.diag(counts)
    precision = np.divide(hits, counts.sum(axis=0), out=np.zeros_like(hits), where=counts.sum(axis=0) > 0)
    recall = np.divide(hits, counts.sum(axis=1), out=np.zeros_like(hits), where=counts.sum(axis=1) > 0)
    return precision, recall


def _color(model: str) -> str:
    """Color fijo de cada red, el mismo que en las curvas del póster."""
    return P.NETWORK_STYLES.get(model, (TEXT_SECONDARY, "o"))[0]


def _figure(n_rows: int, n_cols: int, width: float, height: float, **kwargs: Any) -> tuple[Figure, np.ndarray]:
    """Crea una grilla de paneles con el fondo del póster (``axes`` siempre 2D)."""
    return plt.subplots(n_rows, n_cols, figsize=(width, height), facecolor=SURFACE, squeeze=False, **kwargs)


# --- Antes vs después ---


def plot_before_after(
    before: Scores,
    after: Scores,
    labels: tuple[str, str] = ("mejor previo", "candidato final"),
    title: str = "Candidato final vs mejor corrida previa",
) -> Figure:
    """Barras del F1 macro de cada tarea: mejor resultado previo contra el candidato final.

    Parameters
    ----------
    before : Scores
        Mejor F1 previo por (modelo, tarea) (ver ``best_previous``).
    after : Scores
        F1 del candidato por (modelo, tarea) (ver ``control_scores``).
    labels : tuple[str, str]
        Leyenda de las barras de antes y de después.
    title : str
        Título de la figura.

    Returns
    -------
    Figure
        Un panel por modelo, con el valor y la diferencia de cada tarea.
    """
    fig, axes = _figure(1, len(MODELS), 11, 4.2, sharey=True)
    y = np.arange(len(P.TASKS))
    for ax, model in zip(axes[0], MODELS):
        prev = [before.get((model, t), np.nan) for t in P.TASKS]
        new = [after.get((model, t), np.nan) for t in P.TASKS]
        ax.barh(y + 0.2, prev, height=0.38, color=SERIES_COLORS[7], label=labels[0])
        ax.barh(y - 0.2, new, height=0.38, color=SERIES_COLORS[0], label=labels[1])
        for i, (a, b) in enumerate(zip(prev, new)):
            ax.text(b + 0.01, i - 0.2, f"{b:.3f}  ({b - a:+.3f})", va="center", fontsize=9, color=TEXT_PRIMARY)
            ax.text(a + 0.01, i + 0.2, f"{a:.3f}", va="center", fontsize=8, color=TEXT_SECONDARY)
        ax.set_yticks(y, [P.TASK_LABELS[t] for t in P.TASKS])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("F1 macro", color=TEXT_SECONDARY, fontsize=9)
        ax.set_title(P.label(model), loc="left", fontsize=11, color=TEXT_PRIMARY)
        P._style(ax)
    P._title(fig, title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper right", ncol=2)
    return fig


# --- Curvas de aprendizaje ---


def task_curves(model_report: Report, task: str) -> tuple[list[int], list[float], list[float]]:
    """F1 macro de una tarea en train y test por época.

    Parameters
    ----------
    model_report : Report
        Reporte de un modelo (con ``history``).
    task : str
        Nombre de la tarea.

    Returns
    -------
    epochs : list[int]
        Épocas.
    train : list[float]
        F1 macro en train de cada época.
    test : list[float]
        F1 macro en test de cada época.
    """
    history = model_report["history"]
    epochs = [h["epoch"] for h in history]
    return epochs, [h["score"][task] for h in history], [h["test_score"][task] for h in history]


def _task_model(report: Report, task: str, model: str) -> Report | None:
    """Reporte del modelo que resuelve esa tarea (las redes conjuntas están en los dos problemas)."""
    for models in report["problems"].values():
        model_report = models.get(model)
        if model_report and task in (model_report.get("test") or {}):
            return model_report
    return None


def _curve_panel(ax: Axes, report: Report, task: str) -> None:
    """Dibuja train (punteado) y test (lleno) de cada modelo en una tarea, marcando la época elegida."""
    all_epochs: list[int] = []
    for model in MODELS:
        model_report = _task_model(report, task, model)
        if not model_report:
            continue
        epochs, train, test = task_curves(model_report, task)
        all_epochs = max(all_epochs, epochs, key=len)  # los modelos pueden tener distinta cantidad de épocas
        color = _color(model)
        ax.plot(epochs, test, color=color, linewidth=2, marker="o", markersize=4, label=f"{P.label(model)} · test")
        ax.plot(epochs, train, color=color, linewidth=1.5, linestyle="--", label=f"{P.label(model)} · train")
        best = model_report["best_epoch"]
        ax.plot(best, test[epochs.index(best)], "o", color=color, markersize=10, markeredgecolor=SURFACE)
    P._style(ax, grid_axis="y")
    ax.set_xticks(all_epochs)
    ax.set_xlabel("época", color=TEXT_SECONDARY)
    ax.set_title(P.TASK_LABELS[task], loc="left", fontsize=11, color=TEXT_PRIMARY)


def plot_train_test_curves(report: Report) -> Figure:
    """F1 macro en train y test por época, un panel por tarea, los dos modelos juntos.

    Parameters
    ----------
    report : Report
        Reporte del candidato final.

    Returns
    -------
    Figure
        Tres paneles. El punto grande es la época con la que se quedó cada red: en la conjunta
        se eligió con el promedio de las 3 tareas, así que puede no ser el máximo de cada panel.
    """
    fig, axes = _figure(1, len(P.TASKS), 15, 4.4)
    for ax, task in zip(axes[0], P.TASKS):
        _curve_panel(ax, report, task)
    axes[0, 0].set_ylabel("F1 macro", color=TEXT_SECONDARY)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    P._title(fig, "Curvas de aprendizaje: train (punteado) vs test (lleno)")
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper right", ncol=4)
    return fig


def _network_runs(report: Report) -> list[tuple[str, Report]]:
    """Cada red entrenada una sola vez: la conjunta aparece en los dos problemas con el mismo historial."""
    runs, seen = [], set()
    for problem, models in report["problems"].items():
        for model in MODELS:
            model_report = models.get(model)
            if not model_report or not model_report.get("history") or (model in seen and model.startswith("joint")):
                continue
            seen.add(model)
            name = P.label(model) if model.startswith("joint") else f"{P.label(model)} · {P.PROBLEM_LABELS[problem]}"
            runs.append((name, model_report))
    return runs


def plot_loss_curves(report: Report) -> Figure:
    """Loss de train y test por época de cada red entrenada.

    Parameters
    ----------
    report : Report
        Reporte del candidato final.

    Returns
    -------
    Figure
        Un panel por red. Las escalas no se comparan entre paneles: la loss de la red conjunta
        suma las 3 tareas.
    """
    runs = _network_runs(report)
    fig, axes = _figure(1, len(runs), 5 * len(runs), 4)
    for ax, (name, model_report) in zip(axes[0], runs):
        history = model_report["history"]
        epochs = [h["epoch"] for h in history]
        ax.plot(epochs, [h["loss"] for h in history], color=SERIES_COLORS[0], linestyle="--", label="train")
        ax.plot(epochs, [h["test_loss"] for h in history], color=SERIES_COLORS[1], marker="o", label="test")
        ax.axvline(model_report["best_epoch"], color=TEXT_SECONDARY, linewidth=0.8, linestyle=":")
        P._style(ax, grid_axis="y")
        ax.set_xticks(epochs)
        ax.set_xlabel("época", color=TEXT_SECONDARY)
        ax.set_title(name, loc="left", fontsize=11, color=TEXT_PRIMARY)
    axes[0, 0].legend(frameon=False, fontsize=9)
    P._title(fig, "Loss por época (la línea punteada vertical es la época elegida)")
    return fig


# --- Precisión y recall ---


def _pr_panel(ax: Axes, tasks_by_model: P.ModelTasks, task: str) -> None:
    """Precisión (barra llena) y recall (barra rayada) de cada clase de una tarea, por modelo."""
    names = P.CLASS_LABELS[task]
    x = np.arange(len(names))
    width = 0.8 / (2 * len(MODELS))
    for i, model in enumerate(MODELS):
        if task not in tasks_by_model.get(model, {}):
            continue
        precision, recall = precision_recall(tasks_by_model[model][task]["confusion"])
        offset = (2 * i - len(MODELS)) * width + width / 2
        ax.bar(x + offset, precision, width, color=_color(model), label=f"{P.label(model)} · precisión")
        ax.bar(x + offset + width, recall, width, color=_color(model), alpha=0.45, hatch="//",
               edgecolor=SURFACE, label=f"{P.label(model)} · recall")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    P._style(ax, grid_axis="y")
    ax.set_title(P.TASK_LABELS[task], loc="left", fontsize=11, color=TEXT_PRIMARY)


def plot_precision_recall(report: Report, split: str = "control") -> Figure:
    """Precisión y recall por clase de los dos modelos, un panel por tarea.

    Muestra el efecto de ``class_weight_power``: con un peso de clase alto, las clases raras
    tienen recall mucho mayor que precisión (se predicen de más).

    Parameters
    ----------
    report : Report
        Reporte del candidato final.
    split : str, default="control"
        ``"test"`` o ``"control"``.

    Returns
    -------
    Figure
        Tres paneles, uno por tarea.
    """
    tasks_by_model = P.model_tasks(report, split)
    fig, axes = _figure(1, len(P.TASKS), 16, 4.4, gridspec_kw={"width_ratios": [4, 2, 4]})
    for ax, task in zip(axes[0], P.TASKS):
        _pr_panel(ax, tasks_by_model, task)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    P._title(fig, f"Precisión vs recall por clase ({P.SPLIT_LABELS[split]})")
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper right", ncol=4)
    return fig


# --- Matrices de confusión ---


def plot_confusions_side_by_side(report: Report, split: str = "control") -> Figure:
    """Matrices de confusión normalizadas por fila: una fila por tarea, una columna por modelo.

    Parameters
    ----------
    report : Report
        Reporte del candidato final.
    split : str, default="control"
        ``"test"`` o ``"control"``.

    Returns
    -------
    Figure
        La diagonal de cada matriz es el recall de cada clase.
    """
    tasks_by_model = P.model_tasks(report, split)
    fig, axes = _figure(len(P.TASKS), len(MODELS), 4.6 * len(MODELS), 4.4 * len(P.TASKS))
    for row, task in enumerate(P.TASKS):
        for col, model in enumerate(MODELS):
            counts = np.array(tasks_by_model[model][task]["confusion"])
            title = f"{P.label(model)} · {P.TASK_LABELS[task]}"
            P.plot_confusion_counts(axes[row, col], title, task, counts, P.CLASS_LABELS[task])
    P._title(fig, f"Matrices de confusión ({P.SPLIT_LABELS[split]})")
    return fig


TEXT_SCALE = 1.8  # cuánto se agranda el texto de las matrices de un solo modelo


def enlarge_text(fig: Figure, scale: float = TEXT_SCALE) -> None:
    """Multiplica por ``scale`` el tamaño de todo el texto de la figura y reacomoda los paneles.

    Parameters
    ----------
    fig : Figure
        Figura ya dibujada (con título y, si hace falta, pie).
    scale : float, default=TEXT_SCALE
        Factor de agrandamiento.
    """
    from matplotlib.text import Text

    for text in fig.findobj(Text):
        text.set_fontsize(text.get_fontsize() * scale)


POSTER_BACKGROUND = "#ffffff"  # fondo por defecto de Google Slides, donde va el póster


def set_background(fig: Figure, color: str = POSTER_BACKGROUND) -> None:
    """Pinta el fondo de la figura, de los paneles y la separación entre celdas de las matrices.

    Parameters
    ----------
    fig : Figure
        Figura ya dibujada.
    color : str, default=POSTER_BACKGROUND
        Color de fondo.
    """
    fig.set_facecolor(color)
    for ax in fig.axes:
        ax.set_facecolor(color)
        ax.grid(which="minor", color=color, linewidth=2)


def save_poster_figure(fig: Figure, out_dir: str | Path, name: str) -> None:
    """Guarda la figura como PNG y PDF respetando su color de fondo.

    Parameters
    ----------
    fig : Figure
        Figura.
    out_dir : str or Path
        Carpeta de salida.
    name : str
        Nombre del archivo, sin extensión.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=200, facecolor=fig.get_facecolor())
    fig.savefig(out_dir / f"{name}.pdf", facecolor=fig.get_facecolor())


def plot_confusions_model(
    report: Report, model: str, split: str = "control", text_scale: float = TEXT_SCALE
) -> Figure:
    """Matrices de confusión normalizadas por fila de un solo modelo, una por tarea.

    Parameters
    ----------
    report : Report
        Reporte del candidato final.
    model : str
        Nombre del modelo (por ejemplo ``"joint_conv_bilstm"``).
    split : str, default="control"
        ``"test"`` o ``"control"``.
    text_scale : float, default=TEXT_SCALE
        Factor de agrandamiento del texto.

    Returns
    -------
    Figure
        Tres paneles en fila; la diagonal de cada matriz es el recall de cada clase.
    """
    tasks = P.model_tasks(report, split)[model]
    fig, axes = _figure(1, len(P.TASKS), 6 * len(P.TASKS), 6, gridspec_kw={"width_ratios": [4, 2.6, 4]})
    for ax, task in zip(axes[0], P.TASKS):
        counts = np.array(tasks[task]["confusion"])
        P.plot_confusion_counts(ax, P.TASK_LABELS[task], task, counts, P.CLASS_LABELS[task])
    for ax in axes[0, 1:]:
        ax.set_ylabel("")  # "real" solo en el primer panel
    enlarge_text(fig, text_scale)
    set_background(fig)
    P._title(fig, f"Matrices de confusión · {P.label(model)} ({P.SPLIT_LABELS[split]})")
    fig._suptitle.set_fontsize(fig._suptitle.get_fontsize() * text_scale)
    fig.tight_layout()
    return fig


def plot_confusions_model_by_problem(
    report: Report, model: str, split: str = "control", text_scale: float = TEXT_SCALE
) -> Figure:
    """Como ``plot_confusions_model``, pero con una fila por problema.

    Arriba la capitalización (centrada) y abajo las dos tareas de puntuación.

    Parameters
    ----------
    report : Report
        Reporte del candidato final.
    model : str
        Nombre del modelo (por ejemplo ``"joint_conv_bilstm"``).
    split : str, default="control"
        ``"test"`` o ``"control"``.
    text_scale : float, default=TEXT_SCALE
        Factor de agrandamiento del texto.

    Returns
    -------
    Figure
        Fila 1: capitalización. Fila 2: puntuación inicial y final.
    """
    tasks = P.model_tasks(report, split)[model]
    fig = plt.figure(figsize=(12.5, 12.5), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 1, hspace=0.3)
    top = grid[0].subgridspec(1, 3, width_ratios=[1.6, 4, 1.6])
    bottom = grid[1].subgridspec(1, 2, width_ratios=[2.6, 4], wspace=0.35)
    axes = {
        "capitalizacion": fig.add_subplot(top[0, 1]),
        "punt_inicial": fig.add_subplot(bottom[0, 0]),
        "punt_final": fig.add_subplot(bottom[0, 1]),
    }
    for task, ax in axes.items():
        counts = np.array(tasks[task]["confusion"])
        P.plot_confusion_counts(ax, P.TASK_LABELS[task], task, counts, P.CLASS_LABELS[task])
    axes["punt_final"].set_ylabel("")  # "real" solo en el primer panel de cada fila
    enlarge_text(fig, text_scale)
    set_background(fig)
    P._title(fig, f"Matrices de confusión · {P.label(model)} ({P.SPLIT_LABELS[split]})")
    fig._suptitle.set_fontsize(fig._suptitle.get_fontsize() * text_scale)
    fig.tight_layout()
    return fig
