from src.data import download_dataset, preprocess_dataset, split_dataset_dev_control
from src.features import (
    ClassicFeatureExtractor,
    EmbeddingsExtractor,
    AverageEmbeddingsExtractor,
)
from src.models import train_models, save_models

if __name__ == '__main__':

    try:
        # [1] Descarga localmente el/los datasets a utilizar
        download_dataset()

        # [2] Genera las instancias de torch.util.data.Dataset en base a el/los 
        # datasets descargados en local y el tipo de extractor de embeddings
        for feature_extractor in (
            ClassicFeatureExtractor(),
            AverageEmbeddingsExtractor(),
            EmbeddingsExtractor()
        ):
            (
                punctuation_dataset,
                capitalization_dataset
            ) = preprocess_dataset(feature_extractor)

            for dataset in (capitalization_dataset, punctuation_dataset):
                # 3
                development_set, control_set = split_dataset_dev_control(dataset)

                # 4
                models = train_models(development_set)
                save_models(models)

                # 5
                # evaluate_performance(models, control_set)
                pass
                        
    except Exception as e:
        print(e)
