from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.constants import (
    BATCH_SIZE,
    DATASET_PATH,
    CONTROL_RATIO,
    RANDOM_SEED,
    CAPITALIZATION_DATASET_SIZE,
    PUNCTUATION_DATASET_SIZE,
    TEST_RATIO,
)
from src.data.base import collate_batch
from src.data.punctuation_dataset import PunctuationDataset
from src.data.capitalization_dataset import CapitalizationDataset


def preprocess_dataset(
    punctuation_size: int = PUNCTUATION_DATASET_SIZE,
    capitalization_size: int = CAPITALIZATION_DATASET_SIZE
) -> tuple[Dataset, Dataset]:
    """
    A partir de los datasets locales crea las instancias del punctuation dataset y 
    capitalization dataset, con la cantidad de oraciones pedida para cada uno.

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

    punctuation_dataset = PunctuationDataset(DATASET_PATH, punctuation_size)
    capitalization_dataset = CapitalizationDataset(DATASET_PATH, capitalization_size)

    return punctuation_dataset, capitalization_dataset

def split_dataset_dev_control(dataset: Dataset) -> tuple[Dataset, Dataset]:
    """Divide el dataset en Train/Test devolviendo los Datasets"""

    total_size = len(dataset)
    control_size = int(CONTROL_RATIO * total_size)
    development_size = total_size - control_size

    development_dataset, control_dataset = random_split(
        dataset, 
        [development_size, control_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    return development_dataset, control_dataset


def split_train_test(development_dataset: Dataset) -> tuple[Dataset, Dataset]:
    """Divide el dataset de desarrollo en train y test. Con la misma semilla, todos los
    modelos se entrenan y evalúan sobre las mismas oraciones"""
    train_dataset, test_dataset = random_split(
        development_dataset,
        [1 - TEST_RATIO, TEST_RATIO],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    return train_dataset, test_dataset


def build_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
    """DataLoader por batches de oraciones, con las etiquetas rellenadas con IGNORE_INDEX"""
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, collate_fn=collate_batch)
