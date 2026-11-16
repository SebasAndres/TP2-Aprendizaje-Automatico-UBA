"""Gráficos del póster a partir del reporte de una corrida (ver src/poster/data.py).

Cada modelo junta sus tareas de los dos problemas: capitalización (del problema de
capitalización) y puntuación inicial y final (del de puntuación). Los colores siguen la paleta
de src/training/plots.py: cada tarea tiene siempre el mismo color, y los modelos se nombran en
el eje (son más que los colores que se distinguen bien).
"""
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.data.capitalization_dataset import CAPITALIZATION_NAMES
from src.training.plots import (
    GRID, SEQUENTIAL, SERIES_COLORS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, plot_confusion_counts,
)

Report = dict[str, Any]
ModelTasks = dict[str, dict[str, dict[str, Any]]]  # modelo -> tarea -> {"f1_macro", "confusion"}

TASKS = ["capitalizacion", "punt_inicial", "punt_final"]
TASK_LABELS = {"capitalizacion": "Capitalización", "punt_inicial": "Puntuación inicial", "punt_final": "Puntuación final"}
TASK_COLORS = {task: SERIES_COLORS[i] for i, task in enumerate(TASKS)}
# Orden fijo de los modelos en todos los gráficos, con su nombre para el póster
MODEL_LABELS = {
    "classic_features": "RF atributos",
    "classic_features_dictionary": "RF atributos + diccionario",
    "classic_embeddings": "RF embeddings",
    "standard_rnn": "RNN",
    "bidirectional_rnn": "BiRNN",
    "joint_rnn": "RNN conjunta",
    "bidirectional_joint_rnn": "BiRNN conjunta",
    "joint_lstm": "LSTM conjunta",
    "bidirectional_joint_lstm": "BiLSTM conjunta",
}
SPLIT_LABELS = {"test": "test", "control": "control"}
PROBLEM_LABELS = {"CapitalizationProblem": "capitalización", "PunctuationProblem": "puntuación"}
# Nombres de las clases para el póster (un "," o un "." sueltos casi no se ven)
CLASS_LABELS = {
    "capitalizacion": CAPITALIZATION_NAMES,
    "punt_inicial": ["nada", "apertura ¿"],
    "punt_final": ["nada", "coma ,", "punto .", "cierre ?"],
}


# --- Datos ---

def model_tasks(report: Report, split: str) -> ModelTasks:
    """Las tareas de cada modelo en ese split, juntando los dos problemas, en el orden de MODEL_LABELS"""
    merged: ModelTasks = {}
    for models in report["problems"].values():
        for model, model_report in models.items():
            if model_report.get(split):
                merged.setdefault(model, {}).update(model_report[split])
    order = [model for model in MODEL_LABELS if model in merged] + sorted(set(merged) - set(MODEL_LABELS))
    return {model: merged[model] for model in order}


def label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def per_class_f1(confusion: list[list[int]]) -> np.ndarray:
    """F1 de cada clase a partir de la matriz de confusión (0 si no hay datos)"""
    counts = np.asarray(confusion, dtype=float)
    hits = np.diag(counts)
    precision = hits / np.maximum(counts.sum(axis=0), 1)
    recall = hits / np.maximum(counts.sum(axis=1), 1)
    return np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(hits), where=precision + recall > 0)


def class_columns() -> list[tuple[str, int]]:
    """(tarea, clase) de cada columna de los gráficos por clase: 4 + 2 + 4"""
    sizes = {"capitalizacion": 4, "punt_inicial": 2, "punt_final": 4}
    return [(task, c) for task in TASKS for c in range(sizes[task])]


def f1_table(report: Report, split: str) -> str:
    """Tabla de texto con el F1 macro de cada modelo y tarea, y el promedio"""
    lines = [f"{'modelo':<28}" + "".join(f"{TASK_LABELS[t]:>20}" for t in TASKS) + f"{'promedio':>10}"]
    for model, tasks in model_tasks(report, split).items():
        scores = [tasks[t]["f1_macro"] if t in tasks else np.nan for t in TASKS]
        lines.append(f"{label(model):<28}" + "".join(f"{s:>20.3f}" for s in scores) + f"{np.nanmean(scores):>10.3f}")
    return "\n".join(lines)


def best_model(report: Report, problem: str, split: str) -> str:
    """El modelo del problema con mejor F1 macro promedio entre sus tareas"""
    models = report["problems"][problem]
    return max(
        (model for model in models if models[model].get(split)),
        key=lambda model: np.mean([t["f1_macro"] for t in models[model][split].values()]),
    )


# --- Estilo ---

def _style(ax: Axes, grid_axis: str = "x") -> None:
    """Grilla y ejes discretos, sin bordes de arriba ni de la derecha"""
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)


def _title(fig: Figure, title: str) -> None:
    """Título general arriba a la izquierda (tight_layout ya le reserva su lugar)"""
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=TEXT_PRIMARY)
    fig.tight_layout()


def save_figure(fig: Figure, out_dir: str | Path, name: str) -> None:
    """Guarda el gráfico en PNG (para ver) y en PDF (vectorial, para el póster)"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=200, facecolor=SURFACE)
    fig.savefig(out_dir / f"{name}.pdf", facecolor=SURFACE)


# --- Gráficos ---

def plot_f1_by_task(report: Report, split: str) -> Figure:
    """Un panel por tarea con el F1 macro de cada modelo: la comparación general"""
    tasks_by_model = model_tasks(report, split)
    models = list(tasks_by_model)[::-1]  # el primero arriba
    fig, axes = plt.subplots(1, len(TASKS), figsize=(14, 0.45 * len(models) + 1.6), sharey=True, facecolor=SURFACE)
    for ax, task in zip(axes, TASKS):
        scores = [tasks_by_model[m][task]["f1_macro"] if task in tasks_by_model[m] else 0 for m in models]
        bars = ax.barh(range(len(models)), scores, height=0.62, color=TASK_COLORS[task])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8, color=TEXT_SECONDARY)
        ax.set_xlim(0, 1.08)
        ax.set_title(TASK_LABELS[task], loc="left", fontsize=11, color=TEXT_PRIMARY)
        _style(ax)
    axes[0].set_yticks(range(len(models)), [label(m) for m in models], fontsize=9)
    _title(fig, f"F1 macro por tarea ({SPLIT_LABELS[split]})")
    return fig


def _class_matrix(tasks_by_model: ModelTasks) -> np.ndarray:
    """(n_modelos, n_columnas): F1 de cada clase de cada tarea, NaN si el modelo no tiene la tarea"""
    rows = []
    for tasks in tasks_by_model.values():
        f1 = {task: per_class_f1(tasks[task]["confusion"]) for task in tasks}
        rows.append([f1[task][c] if task in f1 else np.nan for task, c in class_columns()])
    return np.array(rows)


def _class_axis(ax: Axes, separator: str, separator_width: float) -> None:
    """Nombres de las clases abajo y de la tarea arriba de cada grupo de columnas, con una
    línea vertical entre tareas"""
    columns = class_columns()
    ax.set_xticks(range(len(columns)), [CLASS_LABELS[task][c] for task, c in columns], fontsize=9)
    start = 0
    for task in TASKS:
        width = sum(1 for t, _ in columns if t == task)
        # x en datos, y en fracción del eje: siempre justo arriba del gráfico
        ax.text(start + (width - 1) / 2, 1.02, TASK_LABELS[task], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=10, color=TEXT_PRIMARY)
        if start:
            ax.axvline(start - 0.5, color=separator, linewidth=separator_width)
        start += width


def plot_class_f1(report: Report, split: str) -> Figure:
    """F1 de cada clase para cada modelo: qué clase le costó más a cada modelo"""
    tasks_by_model = model_tasks(report, split)
    values = _class_matrix(tasks_by_model)
    fig, ax = plt.subplots(figsize=(12, 0.5 * len(values) + 1.8), facecolor=SURFACE)
    ax.imshow(np.ma.masked_invalid(values), cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for (i, j), value in np.ndenumerate(values):
        if not np.isnan(value):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color=SURFACE if value > 0.55 else TEXT_PRIMARY)
    ax.set_xticks(np.arange(-0.5, values.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, len(values)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.set_yticks(range(len(values)), [label(m) for m in tasks_by_model], fontsize=9)
    _class_axis(ax, separator=SURFACE, separator_width=6)
    ax.tick_params(colors=TEXT_SECONDARY, which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _title(fig, f"F1 por clase ({SPLIT_LABELS[split]})")
    return fig


def _pair_panel(ax: Axes, tasks_by_model: ModelTasks, pair: tuple[str, str]) -> None:
    """Barras del F1 por clase de los dos modelos, una al lado de la otra"""
    x = np.arange(len(class_columns()))
    width = 0.38
    for k, (model, offset) in enumerate(zip(pair, (-width / 2, width / 2))):
        values = _class_matrix({model: tasks_by_model[model]})[0]
        bars = ax.bar(x + offset, np.nan_to_num(values), width - 0.04, color=SERIES_COLORS[k], label=label(model))
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7, color=TEXT_SECONDARY)
    ax.set_ylim(0, 1.12)
    _style(ax, grid_axis="y")
    _class_axis(ax, separator=GRID, separator_width=1)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylabel("F1", color=TEXT_SECONDARY)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))


def plot_pairs_by_class(report: Report, split: str, pairs: list[tuple[str, str]], title: str) -> Figure | None:
    """Un panel por par de modelos con el F1 de cada clase. Se saltean los pares incompletos.
    Ej: separada vs conjunta (¿conviene compartir pesos?), unidireccional vs bidireccional"""
    tasks_by_model = model_tasks(report, split)
    pairs = [pair for pair in pairs if all(model in tasks_by_model for model in pair)]
    if not pairs:
        return None
    fig, axes = plt.subplots(len(pairs), 1, figsize=(13, 3.2 * len(pairs) + 0.6), facecolor=SURFACE, squeeze=False)
    for ax, pair in zip(axes[:, 0], pairs):
        _pair_panel(ax, tasks_by_model, pair)
    _title(fig, f"{title} ({SPLIT_LABELS[split]})")
    return fig


def plot_learning_curves(report: Report, problem: str) -> Figure | None:
    """F1 macro promedio en test por época de cada red, solo sobre las tareas del problema (las
    redes conjuntas tienen historial de las 3); el punto grande es la época con la que se quedó.
    None si la corrida no tiene historial (corridas viejas)"""
    models = {m: r for m, r in report["problems"][problem].items() if r.get("history")}
    if not models:
        return None
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    for k, (model, model_report) in enumerate(models.items()):
        epochs = [epoch["epoch"] for epoch in model_report["history"]]
        tasks = list(model_report["test"])  # las tareas del problema
        scores = [np.mean([epoch["test_score"][task] for task in tasks]) for epoch in model_report["history"]]
        ax.plot(epochs, scores, color=SERIES_COLORS[k % len(SERIES_COLORS)], linewidth=2, marker="o",
                markersize=4, label=label(model))
        if model_report.get("best_epoch"):
            best = model_report["best_epoch"]
            ax.plot(best, scores[epochs.index(best)], "o", markersize=9, color=SERIES_COLORS[k % len(SERIES_COLORS)],
                    markeredgecolor=SURFACE, markeredgewidth=2)
    _style(ax, grid_axis="y")
    ax.set_xticks(epochs)  # épocas enteras
    ax.set_xlabel("época", color=TEXT_SECONDARY)
    ax.set_ylabel("F1 macro promedio (test)", color=TEXT_SECONDARY)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    _title(fig, f"Curvas de aprendizaje · {PROBLEM_LABELS.get(problem, problem)}")
    return fig


def plot_confusions(report: Report, split: str, models: dict[str, str]) -> Figure:
    """Matrices de confusión normalizadas por fila (la diagonal es el recall) de un modelo por
    problema. models: problema -> modelo"""
    panels = [(problem, model, task, confusion["confusion"])
              for problem, model in models.items()
              for task, confusion in report["problems"][problem][model][split].items()]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.4), facecolor=SURFACE, squeeze=False)
    for ax, (_, model, task, counts) in zip(axes[0], panels):
        plot_confusion_counts(ax, f"{label(model)} · {TASK_LABELS[task]}", task, np.array(counts), CLASS_LABELS[task])
    _title(fig, f"Matrices de confusión del mejor modelo de cada problema ({SPLIT_LABELS[split]})")
    return fig
