from src.data import download_dataset, preprocess_dataset, split_dataset_dev_control
from src.features import (
    ClassicFeatureExtractor,
    EmbeddingsExtractor,
    AverageEmbeddingsExtractor,
)
from src.models import train_models, save_models
from src.problem import CapitalizationProblem, PunctuationProblem

if __name__ == '__main__':

    try:
        # Descarga localmente el/los datasets a utilizar
        download_dataset()

        # Por cada tipo de Feature Extractor a probar
        for feature_extractor in (
            ClassicFeatureExtractor(),
            AverageEmbeddingsExtractor(),
            EmbeddingsExtractor()
        ):

            # Genera las instancias de torch.util.data.Dataset en base a el/los 
            # datasets descargados en local y el tipo de extractor de embeddings
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
                # Dividimos datos
                development_set, control_set = split_dataset_dev_control(dataset)

                # Entrenamos modelos
                models = train_models(development_set, feature_extractor, problem)
                save_models(models)

                # Elegimos el modelo ganador
                ...

                # Evaluamos performance sobre control para report 
                # problem.metric.score(winner_preds, control_set[y])
                pass
                        
    except Exception as e:
        print(e)
