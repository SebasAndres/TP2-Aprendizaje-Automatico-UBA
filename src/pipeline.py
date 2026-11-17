import logging
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from typing import Any

from src.constants import (
    BATCH_SIZE,
    CAPITALIZATION_DATASET_SIZE,
    CLASS_WEIGHT_POWER,
    HIDDEN_DIM,
    PUNCTUATION_DATASET_SIZE,
    RANDOM_SEED,
)
from src.data import preprocess_dataset, split_dataset_dev_control
from src.data.samples import add_window_args, window_from_args
from src.training import (
    Experiment,
    build_experiments,
    evaluate_on_control,
    create_run_directory,
    problem_dir,
    train_models,
)
from src.training import neural
from src.training.report import data_report, experiment_report, save_report
from src.training.result import format_score
from src.problem import CapitalizationProblem, PunctuationProblem

logger = logging.getLogger("src.pipeline")


def overall_score(score: dict[str, float]) -> float:
    """Resume el F1 de cada tarea en un número para comparar experimentos: el promedio.
    Un experimento sin score (red sin implementar) queda último"""
    if not score:
        return float("-inf")
    return sum(score.values()) / len(score)


def parse_args() -> Namespace:
    """Cantidad de instancias por fuente y largo de cada una; por defecto, los de src/constants.py"""
    parser = ArgumentParser(description="Entrena y compara los modelos de cada problema")
    parser.add_argument(
        "--capitalization-size",
        type=int,
        default=CAPITALIZATION_DATASET_SIZE,
        help=f"instancias de cada fuente para capitalización (default: {CAPITALIZATION_DATASET_SIZE})",
    )
    parser.add_argument(
        "--punctuation-size",
        type=int,
        default=PUNCTUATION_DATASET_SIZE,
        help=f"instancias de cada fuente para puntuación (default: {PUNCTUATION_DATASET_SIZE})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=neural.N_EPOCHS,
        help=f"épocas de entrenamiento de las redes (default: {neural.N_EPOCHS})",
    )
    parser.add_argument(
        "--experiments",
        type=lambda s: s.split(","),
        default=None,
        help="experimentos a entrenar, separados por coma, ej: joint_lstm,bidirectional_rnn (default: todos)",
    )
    add_window_args(parser)
    return parser.parse_args()


def select_experiments(experiments: list[Experiment], names: list[str] | None) -> list[Experiment]:
    """Los experimentos pedidos, en el orden de build_experiments (todos si names es None)"""
    if names is None:
        return experiments
    unknown = set(names) - {experiment.name for experiment in experiments}
    if unknown:
        available = [experiment.name for experiment in experiments]
        raise ValueError(f"Experimentos desconocidos: {sorted(unknown)}. Disponibles: {available}")
    return [experiment for experiment in experiments if experiment.name in names]


def run_config(args: Namespace) -> dict[str, Any]:
    """Con qué datos e hiperparámetros se entrenó la corrida, para el reporte"""
    window = window_from_args(args)
    return {
        "capitalization_size": args.capitalization_size,
        "punctuation_size": args.punctuation_size,
        "experiments": args.experiments,  # None = todos
        "window": asdict(window) if window else None,  # None = una oración por instancia
        "n_epochs": neural.N_EPOCHS,
        "learning_rate": neural.LEARNING_RATE,
        "weight_decay": neural.WEIGHT_DECAY,  # el de las redes que no eligen el suyo
        "bidirectional_rnn_weight_decay": {
            type(problem).__name__: problem.bidirectional_rnn_weight_decay
            for problem in (CapitalizationProblem(), PunctuationProblem())
        },
        "grad_clip_norm": neural.GRAD_CLIP_NORM,
        "class_weight_power": CLASS_WEIGHT_POWER,
        "batch_size": BATCH_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "random_seed": RANDOM_SEED,
    }


if __name__ == "__main__":
    args = parse_args()
    neural.N_EPOCHS = args.epochs  # lo leen todas las redes al entrenar, y run_config para el reporte

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("src").setLevel(logging.INFO)

    logger.info("Cargando experimentos")
    experiments = select_experiments(build_experiments(), args.experiments)

    # Genera las instancias de torch.util.data.Dataset en base a el/los
    # datasets descargados en local
    logger.info("Preprocesando datasets")
    (punctuation_dataset, capitalization_dataset) = preprocess_dataset(
        args.punctuation_size, args.capitalization_size, window_from_args(args)
    )

    # Todo lo que guarda esta ejecución va a su propia carpeta: models/<día>__<hora>/
    run_dir = create_run_directory()
    logger.info("Guardando modelos en %s", run_dir)
    report: dict[str, Any] = {"config": run_config(args), "data": {}, "problems": {}}

    # Para cada problema
    # obs: podría ser paralelo, no creo alcance la RAM
    for dataset, problem in [
        (capitalization_dataset, CapitalizationProblem()),
        (punctuation_dataset, PunctuationProblem()),
    ]:
        logger.info("Problema: %s (%d instancias)", type(problem).__name__, len(dataset))

        development_set, control_set = split_dataset_dev_control(dataset)
        report["data"][type(problem).__name__] = data_report(dataset.samples)
        logger.info("Split: desarrollo=%d, control=%d", len(development_set), len(control_set))

        model_dir = problem_dir(run_dir, problem)
        experiment_results = train_models(development_set, experiments, problem, model_dir)

        # Elegimos el modelo ganador
        winning_experiment = max(experiment_results, key=lambda name: overall_score(experiment_results[name].score))
        logger.info(
            "Ganador en test: %s (%s)", winning_experiment, format_score(experiment_results[winning_experiment].score)
        )

        # Evaluamos todos los modelos sobre control: es el número honesto para comparar, porque
        # test ya se usó para elegir la época de cada red y el ganador
        control_results = {}
        for experiment in experiments:
            if experiment_results[experiment.name].model is None:  # red sin implementar
                continue
            logger.info("Evaluando %s sobre control", experiment.name)
            control_results[experiment.name] = evaluate_on_control(
                control_set, experiment, experiment_results[experiment.name], problem, model_dir
            )
        logger.info(
            "Ganador en control: %s (%s)", winning_experiment, format_score(control_results[winning_experiment].score)
        )

        report["problems"][type(problem).__name__] = {
            name: experiment_report(result, control_results.get(name)) for name, result in experiment_results.items()
        }
        logger.info("Métricas guardadas en %s", save_report(run_dir, report))

    logger.info("Pipeline finalizado")
