import logging
from argparse import ArgumentParser, Namespace

from src.constants import CAPITALIZATION_DATASET_SIZE, PUNCTUATION_DATASET_SIZE
from src.data import preprocess_dataset, split_dataset_dev_control
from src.training import build_experiments, evaluate_on_control, train_models
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
    """Cantidad de oraciones a usar de cada dataset; por defecto, las de src/constants.py"""
    parser = ArgumentParser(description="Entrena y compara los modelos de cada problema")
    parser.add_argument("--capitalization-size", type=int, default=CAPITALIZATION_DATASET_SIZE,
                        help=f"oraciones del dataset de capitalización (default: {CAPITALIZATION_DATASET_SIZE})")
    parser.add_argument("--punctuation-size", type=int, default=PUNCTUATION_DATASET_SIZE,
                        help=f"oraciones del dataset de puntuación (default: {PUNCTUATION_DATASET_SIZE})")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("src").setLevel(logging.INFO)

    try:
        logger.info("Cargando experimentos")
        experiments = build_experiments()

        # Genera las instancias de torch.util.data.Dataset en base a el/los
        # datasets descargados en local
        logger.info("Preprocesando datasets")
        (
            punctuation_dataset,
            capitalization_dataset
        ) = preprocess_dataset(args.punctuation_size, args.capitalization_size)

        # Para cada problema
        # obs: podría ser paralelo, no creo alcance la RAM
        for dataset, problem in [
            (capitalization_dataset, CapitalizationProblem()),
            (punctuation_dataset, PunctuationProblem())
        ]:
            logger.info("Problema: %s (%d oraciones)", type(problem).__name__, len(dataset))

            development_set, control_set = split_dataset_dev_control(dataset)
            logger.info("Split: desarrollo=%d, control=%d", len(development_set), len(control_set))

            experiment_results = train_models(development_set, experiments, problem)

            # Elegimos el modelo ganador
            winning_experiment = max(
                experiment_results,
                key=lambda name: overall_score(experiment_results[name].score)
            )
            logger.info(
                "Ganador en test: %s (%s)",
                winning_experiment, format_score(experiment_results[winning_experiment].score)
            )

            # Evaluamos performance sobre control para report
            logger.info("Evaluando %s sobre control", winning_experiment)
            experiment = next(e for e in experiments if e.name == winning_experiment)
            control_result = evaluate_on_control(
                control_set, experiment, experiment_results[winning_experiment], problem
            )
            logger.info(
                "Ganador en control: %s (%s)",
                winning_experiment, format_score(control_result.score)
            )


        logger.info("Pipeline finalizado")

    except Exception as e:
        logger.exception(f"El pipeline falló por {e}")
