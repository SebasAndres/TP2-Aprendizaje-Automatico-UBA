"""Tablas y gráficos de los barridos de `src.regularization`, para los notebooks.

Solo usa json y matplotlib (no importa torch ni el resto de src), así funciona en el kernel
de Colab aunque las dependencias del proyecto estén instaladas con uv en otro entorno.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

REGULARIZATION_DIR = Path("models") / "regularization"


def latest_results(param: str) -> Path:
    """El results.json del barrido más reciente de `param` ('dropout' o 'weight_decay')"""
    runs = sorted(REGULARIZATION_DIR.glob(f"*__{param}/results.json"))
    if not runs:
        raise FileNotFoundError(f"No hay barridos de {param} en {REGULARIZATION_DIR}")
    return runs[-1]


def load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _mean(score: dict[str, float]) -> float:
    return sum(score.values()) / len(score)


def _best_gap(run: dict[str, Any]) -> float:
    """F1 promedio de train menos el de test en la mejor época: cuánto sobreajusta"""
    epoch = run["history"][run["best_epoch"] - 1]
    return _mean(epoch["train_f1"]) - _mean(epoch["test_f1"])


def summary_table(report: dict[str, Any]) -> str:
    """Una fila por problema y valor: mejor época, brecha train-test y F1 en test y control"""
    lines = []
    for problem, runs in report["problems"].items():
        tasks = list(runs[0]["control_f1"])
        header = ["valor", "época", "brecha", "test", "control"] + [f"ctrl {t}" for t in tasks]
        lines += [f"\n{problem}", "  ".join(f"{h:>14}" for h in header)]
        for run in runs:
            cells = [f"{run['value']:g}", str(run["best_epoch"]), f"{_best_gap(run):.3f}"]
            cells += [f"{_mean(run['test_f1']):.3f}", f"{_mean(run['control_f1']):.3f}"]
            cells += [f"{run['control_f1'][t]:.3f}" for t in tasks]
            lines.append("  ".join(f"{c:>14}" for c in cells))
    return "\n".join(lines)


def best_value(report: dict[str, Any], problem: str) -> float:
    """El valor con mejor F1 promedio sobre control para ese problema"""
    return max(report["problems"][problem], key=lambda run: _mean(run["control_f1"]))["value"]


def _subplots(report: dict[str, Any], title: str) -> tuple[Figure, list[Any]]:
    n = len(report["problems"])
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    fig.suptitle(title)
    return fig, list(axes[0])


def plot_control_f1(report: dict[str, Any], param: str) -> Figure:
    """F1 macro sobre control de cada tarea (y el promedio) según el valor del hiperparámetro.
    El eje x es categórico: los valores no están equiespaciados (ej: 0, 1e-4, 1e-2)"""
    fig, axes = _subplots(report, f"F1 macro en control según {param}")
    for ax, (problem, runs) in zip(axes, report["problems"].items()):
        labels = [f"{run['value']:g}" for run in runs]
        for task in runs[0]["control_f1"]:
            ax.plot(labels, [run["control_f1"][task] for run in runs], marker="o", label=task)
        ax.plot(labels, [_mean(run["control_f1"]) for run in runs], "k--", marker="s", label="promedio")
        ax.set(title=problem, xlabel=param, ylabel="F1 macro")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return fig


def plot_loss_curves(report: dict[str, Any], param: str) -> Figure:
    """Loss por época de cada valor: train punteada, test llena. Con más regularización la
    loss de test debería dejar de subir en las últimas épocas"""
    fig, axes = _subplots(report, f"Loss por época según {param} (train punteada, test llena)")
    for ax, (problem, runs) in zip(axes, report["problems"].items()):
        for i, run in enumerate(runs):
            epochs = [h["epoch"] for h in run["history"]]
            color = f"C{i}"
            ax.plot(epochs, [h["train_loss"] for h in run["history"]], ":", color=color)
            ax.plot(epochs, [h["test_loss"] for h in run["history"]], "-", color=color, label=f"{run['value']:g}")
        ax.set(title=problem, xlabel="época", ylabel="loss")
        ax.grid(alpha=0.3)
        ax.legend(title=param)
    fig.tight_layout()
    return fig


def plot_gap(report: dict[str, Any], param: str) -> Figure:
    """Brecha de F1 entre train y test en la mejor época: si la regularización funciona, baja"""
    fig, axes = _subplots(report, f"Sobreajuste según {param}: F1 train − F1 test (mejor época)")
    for ax, (problem, runs) in zip(axes, report["problems"].items()):
        labels = [f"{run['value']:g}" for run in runs]
        ax.bar(labels, [_best_gap(run) for run in runs], color="C0")
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set(title=problem, xlabel=param, ylabel="brecha de F1")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig
