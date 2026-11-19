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
    GRID,
    SEQUENTIAL,
    SERIES_COLORS,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    plot_confusion_counts,
    style_panel,
)

Report = dict[str, Any]
ModelTasks = dict[str, dict[str, dict[str, Any]]]  # modelo -> tarea -> {"f1_macro", "confusion"}

TASKS = ["capitalizacion", "punt_inicial", "punt_final"]
TASK_LABELS = {
    "capitalizacion": "Capitalización",
    "punt_inicial": "Puntuación inicial",
    "punt_final": "Puntuación final",
}
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
    "joint_projected_bigru": "Proy. + BiGRU conjunta",
    "joint_residual_stacked_bilstm": "BiLSTM 2 capas + residual",
    "joint_conv_bilstm": "Conv + BiLSTM conjunta",
}
# Color y marcador fijos de cada red en las curvas de aprendizaje: siguen al modelo, no a su
# posición en el ranking. La paleta validada tiene 8 colores; el noveno (marrón) se eligió
# lejos de todos ellos. Con 9 curvas que se cruzan algunos pares igual se parecen (ΔE ~13 en
# OKLab), así que cada red lleva además su propio marcador. Las unidireccionales (que quedan
# abajo) llevan los colores que más se confunden con los de las bidireccionales
NETWORK_STYLES = {
    "standard_rnn": ("#2a78d6", "o"),  # azul
    "joint_rnn": ("#eb6834", "s"),  # naranja
    "joint_lstm": ("#1baf7a", "^"),  # aqua
    "bidirectional_rnn": ("#eda100", "D"),  # amarillo
    "bidirectional_joint_rnn": ("#e87ba4", "v"),  # magenta
    "bidirectional_joint_lstm": ("#008300", "P"),  # verde
    "joint_projected_bigru": ("#6250d6", "X"),  # violeta
    "joint_residual_stacked_bilstm": ("#e34948", "*"),  # rojo
    "joint_conv_bilstm": ("#7a4a1e", "h"),  # marrón
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
    """Junta las tareas de cada modelo en ese split, de los dos problemas.

    Parameters
    ----------
    report : Report
        Reporte de la corrida.
    split : str
        ``"test"`` o ``"control"``.

    Returns
    -------
    ModelTasks
        Tareas de cada modelo, del mejor al peor según el promedio del F1 macro de sus
        tareas (empates: en el orden de ``MODEL_LABELS``).
    """
    merged: ModelTasks = {}
    for models in report["problems"].values():
        for model, model_report in models.items():
            if model_report.get(split):
                merged.setdefault(model, {}).update(model_report[split])
    order = [model for model in MODEL_LABELS if model in merged] + sorted(set(merged) - set(MODEL_LABELS))
    order.sort(key=lambda model: -np.mean([task["f1_macro"] for task in merged[model].values()]))
    return {model: merged[model] for model in order}


def label(model: str) -> str:
    """Devuelve el nombre del modelo para el póster.

    Parameters
    ----------
    model : str
        Nombre del experimento.

    Returns
    -------
    str
        El nombre de ``MODEL_LABELS``, o ``model`` si no tiene.
    """
    return MODEL_LABELS.get(model, model)


def per_class_f1(confusion: list[list[int]]) -> np.ndarray:
    """Calcula el F1 de cada clase a partir de la matriz de confusión.

    Parameters
    ----------
    confusion : list[list[int]]
        Matriz de confusión, fila = clase real, columna = clase predicha.

    Returns
    -------
    np.ndarray of shape (n_clases,)
        F1 de cada clase (0 si no hay datos).
    """
    counts = np.asarray(confusion, dtype=float)
    hits = np.diag(counts)
    precision = hits / np.maximum(counts.sum(axis=0), 1)
    recall = hits / np.maximum(counts.sum(axis=1), 1)
    return np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(hits), where=precision + recall > 0)


def class_columns() -> list[tuple[str, int]]:
    """Lista las columnas de los gráficos por clase: 4 + 2 + 4.

    Returns
    -------
    list[tuple[str, int]]
        ``(tarea, clase)`` de cada columna.
    """
    sizes = {"capitalizacion": 4, "punt_inicial": 2, "punt_final": 4}
    return [(task, c) for task in TASKS for c in range(sizes[task])]


def f1_table(report: Report, split: str) -> str:
    """Arma una tabla de texto con el F1 macro de cada modelo y tarea, y el promedio.

    Parameters
    ----------
    report : Report
        Reporte de la corrida.
    split : str
        ``"test"`` o ``"control"``.

    Returns
    -------
    str
        La tabla, una línea por modelo.
    """
    lines = [f"{'modelo':<28}" + "".join(f"{TASK_LABELS[t]:>20}" for t in TASKS) + f"{'promedio':>10}"]
    for model, tasks in model_tasks(report, split).items():
        scores = [tasks[t]["f1_macro"] if t in tasks else np.nan for t in TASKS]
        lines.append(f"{label(model):<28}" + "".join(f"{s:>20.3f}" for s in scores) + f"{np.nanmean(scores):>10.3f}")
    return "\n".join(lines)


def best_model(report: Report, problem: str, split: str) -> str:
    """Elige el modelo del problema con mejor F1 macro promedio entre sus tareas.

    Parameters
    ----------
    report : Report
        Reporte de la corrida.
    problem : str
        Nombre del problema.
    split : str
        ``"test"`` o ``"control"``.

    Returns
    -------
    str
        Nombre del modelo.
    """
    models = report["problems"][problem]
    return max(
        (model for model in models if models[model].get(split)),
        key=lambda model: np.mean([t["f1_macro"] for t in models[model][split].values()]),
    )


# --- Estilo ---


def _title(fig: Figure, title: str) -> None:
    """Pone el título general arriba a la izquierda (``tight_layout`` ya le reserva su lugar).

    Parameters
    ----------
    fig : Figure
        Figura.
    title : str
        Título general.
    """
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=TEXT_PRIMARY)
    fig.tight_layout()


def _thousands(n: int) -> str:
    """Formatea un entero con punto de miles.

    Parameters
    ----------
    n : int
        Número a formatear.

    Returns
    -------
    str
        Ej: ``9856`` -> ``'9.856'``.
    """
    return f"{n:,}".replace(",", ".")


def _splits_text(splits: dict[str, dict[str, int]]) -> str:
    """Describe el tamaño de los splits de un problema.

    Parameters
    ----------
    splits : dict[str, dict[str, int]]
        ``{"instances", "words"}`` de train, test y control.

    Returns
    -------
    str
        Ej: ``'train 9.856 instancias (317.204 palabras), test 2.464, control 3.080'``.
    """
    train = splits["train"]
    return (
        f"train {_thousands(train['instances'])} instancias ({_thousands(train['words'])} palabras), "
        f"test {_thousands(splits['test']['instances'])}, control {_thousands(splits['control']['instances'])}"
    )


def data_caption(report: Report) -> str:
    """Describe con cuántos datos y épocas se entrenó la corrida, para el pie de cada gráfico.

    Si los dos problemas usaron las mismas instancias se dice una vez; si no, una por problema.

    Parameters
    ----------
    report : Report
        Reporte de la corrida.

    Returns
    -------
    str
        El texto del pie.
    """
    config, data = report["config"], report.get("data", {})
    texts = {problem: _splits_text(splits) for problem, splits in data.items()}
    if len(set(texts.values())) == 1:
        parts = [next(iter(texts.values())) + " por problema"]
    else:
        parts = [f"{PROBLEM_LABELS.get(problem, problem)}: {text}" for problem, text in texts.items()]
    if window := config.get("window"):
        parts.append(f"ventanas de {window['min_size']}–{window['max_size']} palabras")
    if epochs := config.get("n_epochs"):
        parts.append(f"redes: {epochs} épocas")
    return "Datos: " + " · ".join(parts)


def add_caption(fig: Figure, caption: str) -> None:
    """Agrega una línea al pie del gráfico, achicando los paneles para que entre.

    Parameters
    ----------
    fig : Figure
        Figura.
    caption : str
        Texto del pie.
    """
    height = fig.get_figheight()
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + 0.45 / height)
    fig.text(0.01, 0.08 / height, caption, ha="left", va="bottom", fontsize=9, color=TEXT_SECONDARY)


def save_figure(fig: Figure, out_dir: str | Path, name: str) -> None:
    """Guarda el gráfico en PNG (para ver) y en PDF (vectorial, para el póster).

    Parameters
    ----------
    fig : Figure
        Figura a guardar.
    out_dir : str or Path
        Carpeta de salida; se crea si no existe.
    name : str
        Nombre de los archivos, sin extensión.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=200, facecolor=SURFACE)
    fig.savefig(out_dir / f"{name}.pdf", facecolor=SURFACE)


# --- Gráficos ---


def plot_f1_by_task(report: Report, split: str) -> Figure:
    """Dibuja un panel por tarea con el F1 macro de cada modelo: la comparación general.

    Parameters
    ----------
    report : Report
        Reporte de la corrida.
    split : str
        ``"test"`` o ``"control"``.

    Returns
    -------
    Figure
        El gráfico.
    """
    tasks_by_model = model_tasks(report, split)
    models = list(tasks_by_model)[::-1]  # el primero arriba
    fig, axes = plt.subplots(1, len(TASKS), figsize=(14, 0.45 * len(models) + 1.6), sharey=True, facecolor=SURFACE)
    for ax, task in zip(axes, TASKS):
        scores = [tasks_by_model[m][task]["f1_macro"] if task in tasks_by_model[m] else 0 for m in models]
        bars = ax.barh(range(len(models)), scores, height=0.62, color=TASK_COLORS[task])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8, color=TEXT_SECONDARY)
        ax.set_xlim(0, 1.08)
        ax.set_title(TASK_LABELS[task], loc="left", fontsize=11, color=TEXT_PRIMARY)
        style_panel(ax)
    axes[0].set_yticks(range(len(models)), [label(m) for m in models], fontsize=9)
    _title(fig, f"F1 macro por tarea ({SPLIT_LABELS[split]})")
    return fig


def _class_matrix(tasks_by_model: ModelTasks) -> np.ndarray:
    """Arma la matriz de F1 de cada clase de cada tarea, por modelo.

    Parameters
    ----------
    tasks_by_model : ModelTasks
        Tareas de cada modelo.

    Returns
    -------
    np.ndarray of shape (n_modelos, n_columnas)
        F1 de cada clase (ver ``class_columns``); NaN si el modelo no tiene la tarea.
    """
    rows = []
    for tasks in tasks_by_model.values():
        f1 = {task: per_class_f1(tasks[task]["confusion"]) for task in tasks}
        rows.append([f1[task][c] if task in f1 else np.nan for task, c in class_columns()])
    return np.array(rows)


def _class_axis(ax: Axes, separator: str, separator_width: float) -> None:
    """Pone los nombres de las clases abajo y de la tarea arriba de cada grupo de columnas.

    Agrega una línea vertical entre tareas.

    Parameters
    ----------
    ax : Axes
        Panel.
    separator : str
        Color de la línea entre tareas.
    separator_width : float
        Ancho de la línea entre tareas.
    """
    columns = class_columns()
    ax.set_xticks(range(len(columns)), [CLASS_LABELS[task][c] for task, c in columns], fontsize=9)
    start = 0
    for task in TASKS:
        width = sum(1 for t, _ in columns if t == task)
        # x en datos, y en fracción del eje: siempre justo arriba del gráfico
        ax.text(
            start + (width - 1) / 2,
            1.02,
            TASK_LABELS[task],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            color=TEXT_PRIMARY,
        )
        if start:
            ax.axvline(start - 0.5, color=separator, linewidth=separator_width)
        start += width


def plot_class_f1(report: Report, split: str) -> Figure:
    """Dibuja el F1 de cada clase para cada modelo: qué clase le costó más a cada modelo.

    Parameters
    ----------
    report : Report
        Reporte de la corrida.
    split : str
        ``"test"`` o ``"control"``.

    Returns
    -------
    Figure
        El gráfico.
    """
    tasks_by_model = model_tasks(report, split)
    values = _class_matrix(tasks_by_model)
    fig, ax = plt.subplots(figsize=(12, 0.5 * len(values) + 1.8), facecolor=SURFACE)
    ax.imshow(np.ma.masked_invalid(values), cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for (i, j), value in np.ndenumerate(values):
        if not np.isnan(value):
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=SURFACE if value > 0.55 else TEXT_PRIMARY,
            )
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
    """Dibuja las barras del F1 por clase de los dos modelos, una al lado de la otra.

    Parameters
    ----------
    ax : Axes
        Panel.
    tasks_by_model : ModelTasks
        Tareas de cada modelo.
    pair : tuple[str, str]
        Los dos modelos a comparar.
    """
    x = np.arange(len(class_columns()))
    width = 0.38
    for k, (model, offset) in enumerate(zip(pair, (-width / 2, width / 2))):
        values = _class_matrix({model: tasks_by_model[model]})[0]
        bars = ax.bar(x + offset, np.nan_to_num(values), width - 0.04, color=SERIES_COLORS[k], label=label(model))
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7, color=TEXT_SECONDARY)
    ax.set_ylim(0, 1.12)
    style_panel(ax, grid_axis="y")
    _class_axis(ax, separator=GRID, separator_width=1)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylabel("F1", color=TEXT_SECONDARY)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))


def plot_pairs_by_class(report: Report, split: str, pairs: list[tuple[str, str]], title: str) -> Figure | None:
    """Dibuja un panel por par de modelos con el F1 de cada clase.

    Se saltean los pares incompletos. Ej: separada vs conjunta (¿conviene compartir
    pesos?), unidireccional vs bidireccional.

    Parameters
    ----------
    report : Report
        Reporte de la corrida.
    split : str
        ``"test"`` o ``"control"``.
    pairs : list[tuple[str, str]]
        Pares de modelos a comparar.
    title : str
        Título general.

    Returns
    -------
    Figure or None
        El gráfico, o ``None`` si no hay ningún par completo.
    """
    tasks_by_model = model_tasks(report, split)
    pairs = [pair for pair in pairs if all(model in tasks_by_model for model in pair)]
    if not pairs:
        return None
    fig, axes = plt.subplots(len(pairs), 1, figsize=(13, 3.2 * len(pairs) + 0.6), facecolor=SURFACE, squeeze=False)
    for ax, pair in zip(axes[:, 0], pairs):
        _pair_panel(ax, tasks_by_model, pair)
    _title(fig, f"{title} ({SPLIT_LABELS[split]})")
    return fig


def _test_curve(model_report: Report) -> tuple[list[int], list[float]]:
    """Calcula el F1 macro promedio en test de cada época.

    Promedia sobre las tareas del problema (las redes conjuntas tienen historial de las 3).

    Parameters
    ----------
    model_report : Report
        Reporte de un modelo, con ``"test"`` e ``"history"``.

    Returns
    -------
    epochs : list[int]
        Número de cada época.
    scores : list[float]
        F1 macro promedio en test de cada época.
    """
    tasks = list(model_report["test"])
    epochs = [epoch["epoch"] for epoch in model_report["history"]]
    scores = [float(np.mean([epoch["test_score"][task] for task in tasks])) for epoch in model_report["history"]]
    return epochs, scores


def _curve_style(model: str) -> dict[str, Any]:
    """Devuelve el color y el marcador fijos de la red.

    Parameters
    ----------
    model : str
        Nombre del experimento.

    Returns
    -------
    dict[str, Any]
        ``{"color", "marker"}``; gris con círculo si es una red nueva sin estilo asignado.
    """
    color, marker = NETWORK_STYLES.get(model, (TEXT_SECONDARY, "o"))
    return {"color": color, "marker": marker}


def _kept_score(curve: tuple[list[int], list[float]], best_epoch: int | None) -> float:
    """Devuelve el F1 en test de la época con la que se quedó la red.

    Parameters
    ----------
    curve : tuple[list[int], list[float]]
        Épocas y F1 de cada una (ver ``_test_curve``).
    best_epoch : int or None
        Época con la que se quedó; ``None`` usa el máximo.

    Returns
    -------
    float
        El F1 de esa época.
    """
    epochs, scores = curve
    return scores[epochs.index(best_epoch)] if best_epoch else max(scores)


def _problem_curves(report: Report, problem: str) -> dict[str, tuple[tuple[list[int], list[float]], int | None]]:
    """Junta la curva y la época elegida de cada red del problema que tenga historial.

    Parameters
    ----------
    report : Report
        Reporte de la corrida.
    problem : str
        Nombre del problema.

    Returns
    -------
    dict
        ``modelo -> ((épocas, F1), época elegida)``; vacío en corridas viejas.
    """
    return {m: (_test_curve(r), r.get("best_epoch")) for m, r in report["problems"][problem].items() if r.get("history")}


def _draw_learning_curves(ax: Axes, curves: dict[str, tuple[tuple[list[int], list[float]], int | None]]) -> None:
    """Dibuja en ``ax`` las curvas de cada red, de mejor a peor (así quedan en la leyenda).

    Parameters
    ----------
    ax : Axes
        Panel donde dibujar.
    curves : dict
        Curvas de cada red (ver ``_problem_curves``).
    """
    all_epochs: set[int] = set()
    for model in sorted(curves, key=lambda m: _kept_score(*curves[m]), reverse=True):
        (epochs, scores), best = curves[model]
        all_epochs.update(epochs)
        style = _curve_style(model)
        ax.plot(epochs, scores, linewidth=2, markersize=5, label=label(model), **style)
        if best:  # la época con la que se quedó: el mismo marcador, más grande y con borde
            ax.plot(
                best,
                scores[epochs.index(best)],
                markersize=11,
                linestyle="none",
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                **style,
            )
    style_panel(ax, grid_axis="y")
    ax.set_xticks(sorted(all_epochs))  # épocas enteras
    ax.set_xlabel("época", color=TEXT_SECONDARY)


def plot_learning_curves(report: Report, problem: str) -> Figure | None:
    """Dibuja el F1 macro promedio en test por época de cada red.

    Cada red tiene su color y marcador fijos; el marcador grande es la época con la que
    se quedó. La leyenda va de mejor a peor (por el F1 de esa época).

    Parameters
    ----------
    report : Report
        Reporte de la corrida.
    problem : str
        Nombre del problema.

    Returns
    -------
    Figure or None
        El gráfico, o ``None`` si la corrida no tiene historial (corridas viejas).
    """
    curves = _problem_curves(report, problem)
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    _draw_learning_curves(ax, curves)
    ax.set_ylabel("F1 macro promedio (test)", color=TEXT_SECONDARY)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    _title(fig, f"Curvas de aprendizaje · {PROBLEM_LABELS.get(problem, problem)}")
    return fig


def plot_learning_curves_combined(report: Report) -> Figure | None:
    """Dibuja las curvas de aprendizaje de todos los problemas, un panel por problema.

    Cada red tiene el mismo color y marcador en todos los paneles, y hay una sola leyenda
    con todas las redes, de mejor a peor según el promedio (entre los problemas en los que
    aparece) del F1 de la época con la que se quedó.

    Parameters
    ----------
    report : Report
        Reporte de la corrida.

    Returns
    -------
    Figure or None
        El gráfico, o ``None`` si la corrida no tiene historial (corridas viejas).
    """
    by_problem = {p: c for p in report["problems"] if (c := _problem_curves(report, p))}
    if not by_problem:
        return None
    fig, axes = plt.subplots(1, len(by_problem), figsize=(6 * len(by_problem) + 2.5, 4.4), facecolor=SURFACE,
                             squeeze=False)
    kept: dict[str, list[float]] = {}
    for ax, (problem, curves) in zip(axes[0], by_problem.items()):
        _draw_learning_curves(ax, curves)
        ax.set_title(PROBLEM_LABELS.get(problem, problem).capitalize(), loc="left", fontsize=11, color=TEXT_PRIMARY)
        for model, curve in curves.items():
            kept.setdefault(model, []).append(_kept_score(*curve))
    axes[0][0].set_ylabel("F1 macro promedio (test)", color=TEXT_SECONDARY)
    # Una sola leyenda: un handle por red (sin repetir las que están en los dos paneles)
    handles = {}
    for ax in axes[0]:
        for handle, text in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(text, handle)
    order = sorted(kept, key=lambda m: np.mean(kept[m]), reverse=True)
    axes[0][-1].legend(
        [handles[label(m)] for m in order],
        [label(m) for m in order],
        frameon=False,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )
    _title(fig, "Curvas de aprendizaje")
    return fig


def plot_confusions(report: Report, split: str, models: dict[str, str]) -> Figure:
    """Dibuja las matrices de confusión normalizadas por fila de un modelo por problema.

    La diagonal es el recall.

    Parameters
    ----------
    report : Report
        Reporte de la corrida.
    split : str
        ``"test"`` o ``"control"``.
    models : dict[str, str]
        Modelo a mostrar de cada problema.

    Returns
    -------
    Figure
        El gráfico, un panel por tarea.
    """
    panels = [
        (problem, model, task, confusion["confusion"])
        for problem, model in models.items()
        for task, confusion in report["problems"][problem][model][split].items()
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.4), facecolor=SURFACE, squeeze=False)
    for ax, (_, model, task, counts) in zip(axes[0], panels):
        plot_confusion_counts(ax, f"{label(model)} · {TASK_LABELS[task]}", task, np.array(counts), CLASS_LABELS[task])
    _title(fig, f"Matrices de confusión del mejor modelo de cada problema ({SPLIT_LABELS[split]})")
    return fig
