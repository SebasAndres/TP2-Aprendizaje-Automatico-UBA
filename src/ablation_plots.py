"""Gráficos resumen de `src.data_ablation`, a partir del results.csv de una corrida.

- `delta.png`: diferencia de F1 (promedio de las 3 tareas) de cada config contra la primera
  (la de referencia), con barras de error = desvío entre semillas. En modo `sources` hay dos
  barras por config: sobre todo eval y sobre las fuentes que SÍ estaban en train ("otras").
- `tasks.png`: F1 macro de cada tarea por config, un punto por modelo (media ± desvío).
- `per_source_<modelo>.png`: mapa de calor config × fuente de eval con la diferencia de F1
  contra la referencia (azul = mejora, rojo = empeora, gris = igual).

Se regeneran sin reentrenar con:

    uv run python -m src.data_ablation plot models/ablation/<corrida>
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from src.training.plots import GRID, SERIES_COLORS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY

# Divergente: rojo (empeora) <- gris (igual) -> azul (mejora)
DIVERGING = LinearSegmentedColormap.from_list("delta", ["#e34948", "#f0efec", "#2a78d6"])

Results = dict[tuple[str, str, str, str], dict[int, float]]  # (modelo, config, eval, tarea) -> semilla -> f1


def load_results(path: Path) -> tuple[str, list[str], list[str], list[str], Results]:
    """modo, configs, modelos y fuentes (en el orden del csv) y los F1"""
    results: Results = defaultdict(dict)
    configs, models, eval_sets = {}, {}, {}
    mode = ""
    with path.open() as f:
        for row in csv.DictReader(f):
            mode = row["mode"]
            configs[row["config"]] = models[row["model"]] = eval_sets[row["eval_set"]] = None
            results[row["model"], row["config"], row["eval_set"], row["task"]][int(row["seed"])] = float(row["f1"])
    sources = [s for s in eval_sets if s != "all"]
    return mode, list(configs), list(models), sources, results


def _tasks(results: Results) -> list[str]:
    return list(dict.fromkeys(key[3] for key in results))


def mean_f1(results: Results, model: str, config: str, eval_sets: list[str]) -> np.ndarray:
    """F1 promedio de las 3 tareas y de los eval_sets, uno por semilla"""
    tasks = _tasks(results)
    seeds = sorted(results[model, config, eval_sets[0], tasks[0]])
    return np.array(
        [np.mean([results[model, config, e, t][s] for e in eval_sets for t in tasks]) for s in seeds]
    )


def _style(ax: Axes, grid_axis: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)


def _finish(fig: plt.Figure, title: str, subtitle: str, path: Path) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=TEXT_PRIMARY)
    fig.text(0.01, 0.005, subtitle, ha="left", va="bottom", fontsize=8.5, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.03, 1, 1 - 0.45 / fig.get_figheight()))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_delta(mode: str, configs: list[str], models: list[str], sources: list[str], results: Results, out: Path):
    """Barras horizontales: Δ F1 de cada config contra la referencia, un panel por modelo"""
    baseline, others = configs[0], configs[1:]
    groups = [("todo eval", lambda c: ["all"])]
    if mode == "sources":
        groups.append(("fuentes que quedaron en train", lambda c: [s for s in sources if f"-{s}" != c]))
    height = 0.8 / len(groups)

    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 0.55 * len(others) + 1.8),
                             facecolor=SURFACE, squeeze=False, sharey=True, sharex=True)
    for ax, model in zip(axes[0], models):
        y = np.arange(len(others))
        for g, (name, eval_sets_of) in enumerate(groups):
            deltas, errors = [], []
            for config in others:
                eval_sets = eval_sets_of(config)
                diff = mean_f1(results, model, config, eval_sets) - mean_f1(results, model, baseline, eval_sets)
                deltas.append(diff.mean())
                errors.append(diff.std())
            offset = (g - (len(groups) - 1) / 2) * height
            ax.barh(y + offset, deltas, height=height - 0.04, color=SERIES_COLORS[g], label=name,
                    xerr=errors, error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1, "capsize": 2})
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
        ax.set_yticks(y, others)
        ax.invert_yaxis()
        ax.set_title(model, loc="left", fontsize=11, color=TEXT_PRIMARY)
        ax.set_xlabel(f"Δ F1 macro vs {baseline}  (>0 = mejor)", color=TEXT_SECONDARY, fontsize=9)
        _style(ax, "x")
    if len(groups) > 1:
        axes[0][-1].legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1, 1.04), ncol=2)
    n_seeds = len(mean_f1(results, models[0], baseline, ["all"]))
    _finish(fig, f"Δ F1 promedio (cap, punt. inicial, punt. final) vs {baseline}",
            f"barras de error: desvío de la diferencia entre {n_seeds} semillas", out / "delta.png")


def plot_tasks(configs: list[str], models: list[str], results: Results, out: Path):
    """F1 sobre todo eval por tarea y config, un punto por modelo"""
    tasks = _tasks(results)
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 0.45 * len(configs) + 1.8),
                             facecolor=SURFACE, sharey=True)
    y = np.arange(len(configs))
    for ax, task in zip(axes, tasks):
        for m, model in enumerate(models):
            values = [list(results[model, c, "all", task].values()) for c in configs]
            offset = (m - (len(models) - 1) / 2) * 0.25
            ax.errorbar([np.mean(v) for v in values], y + offset, xerr=[np.std(v) for v in values], fmt="o",
                        color=SERIES_COLORS[m], markersize=7, elinewidth=1.5, capsize=2, label=model)
        ax.set_yticks(y, configs)
        ax.invert_yaxis()
        ax.set_title(task, loc="left", fontsize=11, color=TEXT_PRIMARY)
        ax.set_xlabel("F1 macro (todo eval)", color=TEXT_SECONDARY, fontsize=9)
        _style(ax, "x")
    axes[-1].legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    _finish(fig, "F1 macro por tarea", "punto: media entre semillas · barra: desvío", out / "tasks.png")


def plot_per_source(configs: list[str], models: list[str], sources: list[str], results: Results, out: Path):
    """Mapa de calor config × fuente de eval: Δ F1 contra la referencia"""
    baseline = configs[0]
    for model in models:
        columns = ["all"] + sources
        delta = np.array([
            [mean_f1(results, model, c, [e]).mean() - mean_f1(results, model, baseline, [e]).mean() for e in columns]
            for c in configs[1:]
        ])
        limit = max(np.abs(delta).max(), 1e-3)
        fig, ax = plt.subplots(figsize=(1.0 * len(columns) + 2.5, 0.5 * len(configs) + 1.6), facecolor=SURFACE)
        ax.imshow(delta, cmap=DIVERGING, norm=TwoSlopeNorm(0, -limit, limit), aspect="auto")
        for i in range(delta.shape[0]):
            for j in range(delta.shape[1]):
                ax.text(j, i, f"{delta[i, j]:+.3f}", ha="center", va="center", fontsize=8.5, color=TEXT_PRIMARY)
        ax.set_xticks(range(len(columns)), columns, rotation=30, ha="right")
        ax.set_yticks(range(len(configs) - 1), configs[1:])
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("fuente de eval", color=TEXT_SECONDARY, fontsize=9)
        _finish(fig, f"{model}: Δ F1 por fuente de eval vs {baseline}",
                "azul = mejora · rojo = empeora · media entre semillas", out / f"per_source_{model}.png")


def save_summary_plots(run_dir: Path) -> list[Path]:
    mode, configs, models, sources, results = load_results(run_dir / "results.csv")
    if len(configs) < 2:
        return []
    plot_delta(mode, configs, models, sources, results, run_dir)
    plot_tasks(configs, models, results, run_dir)
    plot_per_source(configs, models, sources, results, run_dir)
    return [run_dir / "delta.png", run_dir / "tasks.png"] + [run_dir / f"per_source_{m}.png" for m in models]
