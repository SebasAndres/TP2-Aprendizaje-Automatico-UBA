from pathlib import Path
import torch
from torch.utils.data import Dataset, random_split

from src.constants import (
    DATASET_PATH, 
    CONTROL_RATIO,
    SPLIT_SEED,
)
from src.data.punctuation_dataset import PunctuationDataset
from src.data.capitalization_dataset import CapitalizationDataset
from src.features import TextFeatureExtractor

def download_dataset():
    """
    Acá se descarga el dataset en DATASET_PATH 
    """

    return


def preprocess_dataset() -> tuple[Dataset, Dataset]:
    """
    A partir de los datasets locales crea las instancias del punctuation dataset y 
    capitalization dataset.

    Returns
    ------
    - punctuation_dataset: PunctuationDataset
        El dataset completo para el problema de puntuación 
    - capitalization_datset: CapitalizationDataset
        El dataset completo para el problema de capitalización 
    """

    dataset_file = Path(DATASET_PATH)
    if not dataset_file.is_file():
        raise FileNotFoundError(f"No existe el dataset en {DATASET_PATH}")

    punctuation_dataset = PunctuationDataset(DATASET_PATH)
    capitalization_dataset = CapitalizationDataset(DATASET_PATH)

    return punctuation_dataset, capitalization_dataset

def split_dataset_dev_control(dataset: Dataset) -> tuple[Dataset, Dataset]:
    """Divide el dataset en Train/Test devolviendo los Datasets"""

    total_size = len(dataset)
    control_size = int(CONTROL_RATIO * total_size)
    development_size = total_size - control_size

    development_dataset, control_dataset = random_split(
        dataset, 
        [development_size, control_size],
        generator=torch.Generator().manual_seed(SPLIT_SEED)
    )
    return development_dataset, control_dataset