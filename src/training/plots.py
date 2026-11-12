"""Gráficos de la evolución del entrenamiento de una red, época a época.

Un panel por medida (loss, F1 por tarea, precision/recall de cada tarea), todos con
la época en el eje x. Cada panel tiene su propio eje y: nunca se mezclan escalas.
"""
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo guarda archivos
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.training.neural import EpochMetrics

# Paleta categórica en orden fijo: la clase c siempre usa el color c
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _style_axes(ax: Axes, title: str, epochs: list[int], unit_range: bool) -> None:
    """Ejes y grilla discretos; el título del panel nombra la medida"""
    ax.set_title(title, loc="left", fontsize=11, color=TEXT_PRIMARY)
    ax.set_xlabel("época", color=TEXT_SECONDARY)
    ax.set_xticks(epochs)
    if unit_range:
        ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def _plot_line(ax: Axes, epochs: list[int], values: Sequence[float], color: str, **kwargs) -> None:
    ax.plot(epochs, values, color=color, linewidth=2, marker="o", markersize=5, **kwargs)


def _plot_loss(ax: Axes, history: list[EpochMetrics], epochs: list[int]) -> None:
    _plot_line(ax, epochs, [m.loss for m in history], SERIES_COLORS[0])
    _style_axes(ax, "Loss promedio", epochs, unit_range=False)


def _plot_scores(ax: Axes, history: list[EpochMetrics], epochs: list[int]) -> None:
    tasks = list(history[0].score)
    for i, task in enumerate(tasks):
        _plot_line(ax, epochs, [m.score[task] for m in history], SERIES_COLORS[i], label=task)
    _style_axes(ax, "F1 macro", epochs, unit_range=True)
    if len(tasks) > 1:
        ax.legend(frameon=False, fontsize=9)


def _plot_precision_recall(ax: Axes, history: list[EpochMetrics], epochs: list[int], task: str) -> None:
    """Un color por clase; precision con línea llena y recall con línea punteada"""
    for c in history[0].per_class[task]:
        color = SERIES_COLORS[c]
        _plot_line(ax, epochs, [m.per_class[task][c][0] for m in history], color, label=f"clase {c} · P")
        _plot_line(ax, epochs, [m.per_class[task][c][1] for m in history], color,
                   linestyle="--", markerfacecolor=SURFACE, label=f"clase {c} · R")
    _style_axes(ax, f"{task}: precision (—) y recall (--) por clase", epochs, unit_range=True)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1))


def save_training_plot(history: list[EpochMetrics], path: Path, title: str) -> None:
    """Guarda un PNG con la evolución de la loss y las métricas de cada época"""
    epochs = [m.epoch for m in history]
    tasks = list(history[0].per_class)
    fig, axes = plt.subplots(2 + len(tasks), 1, figsize=(9, 3 * (2 + len(tasks))), facecolor=SURFACE)
    _plot_loss(axes[0], history, epochs)
    _plot_scores(axes[1], history, epochs)
    for ax, task in zip(axes[2:], tasks):
        _plot_precision_recall(ax, history, epochs, task)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
