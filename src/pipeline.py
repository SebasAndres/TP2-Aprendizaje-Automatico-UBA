import logging

from src.data import preprocess_dataset, split_dataset_dev_control
from src.training import build_experiments, train_models, save_models
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

            models = train_models(development_set, experiments, problem)
            logger.info("Guardando modelos")
            save_models(models)

            # Elegimos el modelo ganador
            ...

            # Evaluamos performance sobre control para report
            # problem.metric.score(winner_preds, control_set[y])

        logger.info("Pipeline finalizado")

    except Exception:
        logger.exception("El pipeline falló")
