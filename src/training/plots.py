"""Gráficos que se guardan junto a los modelos.

- `save_training_plot`: evolución de una red época a época. Un panel por medida (loss y F1
  por tarea, de train y de test; precision/recall de cada tarea sobre train); cada panel
  tiene su propio eje y.
- `save_class_metrics_plot`: precision/recall por clase sobre test, para modelos sin épocas.
- `save_confusion_matrices`: una matriz por modelo y tarea, normalizada por fila.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo guarda archivos
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

from src.metrics.base import PerClass, class_name
from src.training.result import EpochMetrics, TrainResult

# Paleta categórica en orden fijo: la clase c siempre usa el color c
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
# Escala secuencial de un solo tono (azul), de claro a oscuro
# El 0 arranca en un celeste muy claro, no en el color de fondo, para que la celda se vea
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#eef4fc", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])


def _finish(fig: plt.Figure, title: str) -> None:
    """Título general arriba a la izquierda, con su propio espacio para no pisar los paneles"""
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))


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


def _plot_train_test(
    ax: Axes, epochs: list[int], train: Sequence[float], test: Sequence[float], color: str, label: str
) -> None:
    """Train con línea llena y test con línea punteada, del mismo color"""
    _plot_line(ax, epochs, train, color, label=f"{label} · train")
    _plot_line(ax, epochs, test, color, linestyle="--", markerfacecolor=SURFACE, label=f"{label} · test")


def _plot_loss(ax: Axes, history: list[EpochMetrics], epochs: list[int]) -> None:
    _plot_train_test(ax, epochs, [m.loss for m in history], [m.test_loss for m in history], SERIES_COLORS[0], "loss")
    _style_axes(ax, "Loss promedio: train (—) y test (--)", epochs, unit_range=False)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))


def _plot_scores(ax: Axes, history: list[EpochMetrics], epochs: list[int]) -> None:
    for i, task in enumerate(history[0].score):
        _plot_train_test(
            ax, epochs, [m.score[task] for m in history], [m.test_score[task] for m in history], SERIES_COLORS[i], task
        )
    _style_axes(ax, "F1 macro: train (—) y test (--)", epochs, unit_range=True)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))


def _plot_precision_recall(ax: Axes, history: list[EpochMetrics], epochs: list[int], task: str) -> None:
    """Un color por clase; precision con línea llena y recall con línea punteada"""
    for c in history[0].per_class[task]:
        color = SERIES_COLORS[c]
        name = class_name(task, c)
        _plot_line(ax, epochs, [m.per_class[task][c][0] for m in history], color, label=f"{name} · P")
        _plot_line(
            ax,
            epochs,
            [m.per_class[task][c][1] for m in history],
            color,
            linestyle="--",
            markerfacecolor=SURFACE,
            label=f"{name} · R",
        )
    _style_axes(ax, f"{task} (train): precision (—) y recall (--) por clase", epochs, unit_range=True)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1))


def save_training_plot(history: list[EpochMetrics], path: Path, title: str, best_epoch: int | None = None) -> None:
    """Guarda un PNG con la evolución de la loss y las métricas de cada época. Si se indica
    `best_epoch` (la época con la que se quedó la red), la marca con una línea punteada"""
    epochs = [m.epoch for m in history]
    tasks = list(history[0].per_class)
    fig, axes = plt.subplots(2 + len(tasks), 1, figsize=(9, 3 * (2 + len(tasks))), facecolor=SURFACE)
    _plot_loss(axes[0], history, epochs)
    _plot_scores(axes[1], history, epochs)
    for ax, task in zip(axes[2:], tasks):
        _plot_precision_recall(ax, history, epochs, task)
    if best_epoch is not None:
        for ax in axes:
            ax.axvline(best_epoch, color=TEXT_SECONDARY, linestyle=":", linewidth=1.2, zorder=0)
        title += f" · mejor época: {best_epoch}"
    _finish(fig, title)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_class_bars(ax: Axes, task: str, classes: Mapping[int, tuple[float, float]]) -> None:
    """Precision y recall de cada clase, una al lado de la otra, con su valor arriba"""
    x = np.arange(len(classes))
    width = 0.38
    for offset, idx, label in ((-width / 2, 0, "precision"), (width / 2, 1, "recall")):
        values = [classes[c][idx] for c in classes]
        bars = ax.bar(x + offset, values, width - 0.04, color=SERIES_COLORS[idx], label=label)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, color=TEXT_SECONDARY)
    _style_axes(ax, task, [], unit_range=True)
    ax.set_ylim(0, 1.12)  # lugar para el valor arriba de las barras que llegan a 1
    ax.set_xticks(x, [class_name(task, c) for c in classes])
    ax.set_xlabel("clase", color=TEXT_SECONDARY)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))


def save_class_metrics_plot(per_class: PerClass, path: Path, title: str) -> None:
    """Guarda un PNG con la precision y el recall de cada clase, un panel por tarea"""
    tasks = list(per_class)
    fig, axes = plt.subplots(len(tasks), 1, figsize=(9, 3.4 * len(tasks)), facecolor=SURFACE, squeeze=False)
    for ax, task in zip(axes[:, 0], tasks):
        _plot_class_bars(ax, task, per_class[task])
    _finish(fig, title)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_confusion(ax: Axes, title: str, task: str, real: np.ndarray, pred: np.ndarray, n_classes: int) -> None:
    plot_confusion_counts(ax, title, task, confusion_matrix(real, pred, labels=range(n_classes)))


def plot_confusion_counts(ax: Axes, title: str, task: str, counts: np.ndarray, names: list[str] | None = None) -> None:
    """Matriz normalizada por fila: cada fila muestra a dónde fueron las palabras de esa clase
    real (la diagonal es el recall). Cada celda indica el porcentaje y la cantidad.
    counts: (n_clases, n_clases), fila = clase real, columna = clase predicha
    names: nombre de cada clase (por defecto, los de class_name)"""
    counts = np.asarray(counts)
    n_classes = len(counts)
    rates = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    ax.imshow(rates, cmap=SEQUENTIAL, vmin=0, vmax=1)
    # Separación de 2px del color de fondo entre celdas
    ax.set_xticks(np.arange(-0.5, n_classes), minor=True)
    ax.set_yticks(np.arange(-0.5, n_classes), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(n_classes):
        for j in range(n_classes):
            color = SURFACE if rates[i, j] > 0.55 else TEXT_PRIMARY
            ax.text(j, i, f"{rates[i, j]:.0%}\n{counts[i, j]}", ha="center", va="center", fontsize=8, color=color)
    names = names or [class_name(task, c) for c in range(n_classes)]
    ax.set_xticks(range(n_classes), names, fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticks(range(n_classes), names, fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlabel("predicha", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("real", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(title, loc="left", fontsize=10, color=TEXT_PRIMARY)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_confusion_matrices(results: Mapping[str, TrainResult], path: Path, title: str) -> None:
    """Guarda un PNG con una fila por modelo y una columna por tarea"""
    evaluated = {name: result for name, result in results.items() if result.test_tasks}
    if not evaluated:
        return
    tasks = list(next(iter(evaluated.values())).test_tasks)
    fig, axes = plt.subplots(
        len(evaluated), len(tasks), figsize=(4.4 * len(tasks), 3.9 * len(evaluated)), facecolor=SURFACE, squeeze=False
    )
    for row, (name, result) in zip(axes, evaluated.items()):
        for ax, task in zip(row, tasks):
            real, pred, n_classes = result.test_tasks[task]
            _plot_confusion(ax, f"{name} · {task}", task, real, pred, n_classes)
    _finish(fig, title)
    fig.savefig(path, dpi=120)
    plt.close(fig)
