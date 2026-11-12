import logging

from src.data import preprocess_dataset, split_dataset_dev_control
from src.training import build_experiments, train_models
from src.problem import CapitalizationProblem, PunctuationProblem

logger = logging.getLogger("src.pipeline")

if __name__ == '__main__':

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
        ) = preprocess_dataset()

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
            winning_experiment = max(experiment_results, key=lambda k: experiment_results[k]["score"])            
            logger.info(
                f"El experimento que mejor performó para TEST fue {winning_experiment} "
                f"performó con un score={experiment_results[winning_experiment]}"
            )

            # Evaluamos performance sobre control para report
            # TODO
            

        logger.info("Pipeline finalizado")

    except Exception as e:
        logger.exception(f"El pipeline falló por {e}")
