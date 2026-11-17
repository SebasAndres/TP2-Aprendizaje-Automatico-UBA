"""Una sola RNN que predice capitalización y puntuación al mismo tiempo.

La RNN se comparte entre las tres tareas y cada tarea tiene su propia "cabeza" lineal:
- capitalización: 4 clases (0, 1, 2, 3)
- puntuación inicial: 2 clases ("", ¿)
- puntuación final: 4 clases ("", ",", ".", "?")

Las redes están en `src/problem/joint/rnn.py` (`JointRNN` y `JointLSTM`, la misma red con
celdas LSTM). Este módulo solo las entrena desde la línea de comandos.

En el pipeline son los experimentos `joint_rnn`, `bidirectional_joint_rnn`, `joint_lstm` y
`bidirectional_joint_lstm`: se entrenan con las 3 tareas sobre las mismas instancias de train
que el resto de los modelos, y para cada problema se evalúan solo sus cabezas. También se
pueden correr solas, con las 3 tareas a la vez:

    uv run python -m src.joint_rnn                      # todas las instancias de cada fuente
    uv run python -m src.joint_rnn --size 30000         # como mucho 30000 instancias por fuente
    uv run python -m src.joint_rnn --bidirectional      # RNN bidireccional
    uv run python -m src.joint_rnn --lstm               # LSTM en lugar de RNN simple
"""

import argparse
import logging

from torch.utils.data import DataLoader, Dataset

from src.constants import CLASS_WEIGHT_POWER
from src.data import (
    JointDataset,
    build_loader,
    load_documents,
    load_samples,
    split_dataset_dev_control,
    split_train_test,
)
from src.data.samples import add_window_args, window_from_args
from src.features import EmbeddingsExtractor
from src.metrics import JointMetric
from src.problem import CapitalizationProblem, PunctuationProblem
from src.problem.joint import create_joint_model, joint_model_name
from src.training.neural import N_EPOCHS, train_neural_model
from src.training.result import TrainResult, format_score
from src.training.train import _file_stem, _save_experiment, create_run_directory, problem_dir

logger = logging.getLogger("src.joint_rnn")


def _loaders(dataset: Dataset) -> tuple[DataLoader, DataLoader]:
    """Crea los loaders de train y test con los mismos splits y semillas que el pipeline.

    Parameters
    ----------
    dataset : Dataset
        El ``JointDataset`` completo.

    Returns
    -------
    train_loader : DataLoader
        Instancias de train, mezcladas.
    test_loader : DataLoader
        Instancias de test.
    """
    development_set, _ = split_dataset_dev_control(dataset)
    train_set, test_set = split_train_test(development_set)
    logger.info("Instancias: train=%d, test=%d", len(train_set), len(test_set))
    return build_loader(train_set, shuffle=True), build_loader(test_set, shuffle=False)


def main() -> None:
    """Entrena una red conjunta desde la línea de comandos y la guarda como el pipeline."""
    parser = argparse.ArgumentParser(description="Entrena una RNN conjunta de capitalización y puntuación")
    parser.add_argument(
        "--size", type=int, default=None, help="usar como mucho N instancias de cada fuente (default: todas)"
    )
    add_window_args(parser)
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=CLASS_WEIGHT_POWER,
        help=f"peso de las clases raras en la loss: 1 = balanceado, 0 = sin pesos (default: {CLASS_WEIGHT_POWER})",
    )
    parser.add_argument("--epochs", type=int, default=N_EPOCHS, help=f"épocas de entrenamiento (default: {N_EPOCHS})")
    parser.add_argument("--bidirectional", action="store_true", help="usar una RNN bidireccional")
    parser.add_argument("--lstm", action="store_true", help="usar celdas LSTM en lugar de la RNN simple")
    parser.add_argument("--no-save", action="store_true", help="no guardar la red entrenada en models/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("src").setLevel(logging.INFO)

    samples = load_samples(load_documents(), args.size, window_from_args(args))
    dataset = JointDataset(samples)
    train_loader, test_loader = _loaders(dataset)
    model = create_joint_model(args.bidirectional, args.lstm)
    name = joint_model_name(args.bidirectional, args.lstm)
    result = train_neural_model(
        train_loader,
        test_loader,
        EmbeddingsExtractor(),
        JointMetric(),
        model,
        n_epochs=args.epochs,
        class_weight_power=args.class_weight_power,
    )
    print(f"\n{name} - F1 macro sobre test: {format_score(result.score)}")
    if not args.no_save:
        save_joint_result(result, name)


def save_joint_result(result: TrainResult, name: str) -> None:
    """Guarda la red como lo hace el pipeline, una copia por problema con el score de sus tareas.

    El archivo queda en ``models/<día>__<hora>/<Problema>/<score>__<name>.pkl``. Así
    ``src.inference`` la encuentra al elegir cualquiera de los dos problemas, y como es
    conjunta predice los dos.

    Parameters
    ----------
    result : TrainResult
        Resultado del entrenamiento, con las 3 tareas.
    name : str
        Nombre del experimento (ver ``joint_model_name``).
    """
    run_dir = create_run_directory()
    for problem in (CapitalizationProblem(), PunctuationProblem()):
        problem_result = result.only_tasks(problem.metric.tasks)
        model_dir = problem_dir(run_dir, problem)
        file_stem = _file_stem(model_dir, name, problem_result)
        _save_experiment(name, problem_result, file_stem, type(problem).__name__)
        print(f"Guardado en {file_stem}.pkl")


if __name__ == "__main__":
    main()
