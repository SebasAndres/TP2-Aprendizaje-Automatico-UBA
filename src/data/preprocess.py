import logging
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from src.constants import (
    BATCH_SIZE,
    DATASET_SOURCES,
    CONTROL_RATIO,
    RANDOM_SEED,
    CAPITALIZATION_DATASET_SIZE,
    PUNCTUATION_DATASET_SIZE,
    TEST_RATIO,
)
from src.data.base import collate_batch
from src.data.punctuation_dataset import PunctuationDataset
from src.data.capitalization_dataset import CapitalizationDataset
from src.data.samples import Sample, WindowSpec, build_samples
from src.data.sources import Document, read_line_sentences, read_srt, read_subtitles, read_ted

logger = logging.getLogger(__name__)

# Lector de cada fuente de DATASET_SOURCES
READERS: dict[str, Callable[[str], list[Document]]] = {
    "tatoeba": read_line_sentences,
    "rioplatense": read_line_sentences,
    "ted": read_ted,
    "mrbeast": read_subtitles,
    "peliculas": read_srt,
    "simuladores": read_srt,
    "okupas": read_srt,
}


def load_documents() -> dict[str, list[Document]]:
    """Lee cada fuente de DATASET_SOURCES con su lector. Una fuente que no está en disco se
    saltea con un warning; si no hay ninguna, falla"""
    documents = {}
    for name, path in DATASET_SOURCES.items():
        if not Path(path).exists():
            logger.warning("Fuente %s: no existe %s, se saltea", name, path)
            continue
        documents[name] = READERS[name](path)
    if not documents:
        raise FileNotFoundError(f"No se encontró ninguna fuente de {list(DATASET_SOURCES.values())}")
    return documents


def load_samples(
    documents: dict[str, list[Document]],
    size: int | None,
    window: WindowSpec | None = WindowSpec(),
) -> list[Sample]:
    """Corta cada fuente en instancias (ventanas o oraciones) y toma las primeras `size` de
    cada una (todas si es None), así ninguna fuente tapa al resto"""
    samples = []
    for name, source_documents in documents.items():
        source_samples = build_samples(source_documents, window)[:size]
        logger.info("Fuente %s: %d instancias", name, len(source_samples))
        samples.extend(source_samples)
    return samples


def preprocess_dataset(
    punctuation_size: int = PUNCTUATION_DATASET_SIZE,
    capitalization_size: int = CAPITALIZATION_DATASET_SIZE,
    window: WindowSpec | None = WindowSpec(),
) -> tuple[Dataset, Dataset]:
    """
    A partir de las fuentes locales crea las instancias del punctuation dataset y
    capitalization dataset. Cada fuente aporta como mucho `*_size` instancias, cortadas
    en ventanas según `window` (o una oración por instancia si window es None).

    Returns
    ------
    - punctuation_dataset: PunctuationDataset
        El dataset completo para el problema de puntuación
    - capitalization_datset: CapitalizationDataset
        El dataset completo para el problema de capitalización
    """
    documents = load_documents()
    punctuation_dataset = PunctuationDataset(load_samples(documents, punctuation_size, window))
    capitalization_dataset = CapitalizationDataset(load_samples(documents, capitalization_size, window))

    return punctuation_dataset, capitalization_dataset


def split_dataset_dev_control(dataset: Dataset) -> tuple[Dataset, Dataset]:
    """Divide el dataset en Train/Test devolviendo los Datasets"""

    total_size = len(dataset)
    control_size = int(CONTROL_RATIO * total_size)
    development_size = total_size - control_size

    development_dataset, control_dataset = random_split(
        dataset, [development_size, control_size], generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    return development_dataset, control_dataset


def split_train_test(development_dataset: Dataset) -> tuple[Dataset, Dataset]:
    """Divide el dataset de desarrollo en train y test. Con la misma semilla, todos los
    modelos se entrenan y evalúan sobre las mismas oraciones"""
    train_dataset, test_dataset = random_split(
        development_dataset, [1 - TEST_RATIO, TEST_RATIO], generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    return train_dataset, test_dataset


def dataset_samples(dataset: Dataset) -> list[Sample]:
    """Las instancias (palabras con su puntuación y capitalización originales) de un dataset,
    en el mismo orden que dataset[i], atravesando los Subset que generan los splits"""
    if isinstance(dataset, Subset):
        samples = dataset_samples(dataset.dataset)
        return [samples[i] for i in dataset.indices]
    return dataset.samples


def build_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
    """DataLoader por batches de oraciones, con las etiquetas rellenadas con IGNORE_INDEX"""
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, collate_fn=collate_batch)
