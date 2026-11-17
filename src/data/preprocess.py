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
    """Lee cada fuente de ``DATASET_SOURCES`` con su lector.

    Una fuente que no está en disco se saltea con un warning.

    Returns
    -------
    dict[str, list[Document]]
        Los documentos de cada fuente encontrada, por nombre de fuente.

    Raises
    ------
    FileNotFoundError
        Si no se encontró ninguna fuente.
    """
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
    """Corta cada fuente en instancias y toma las primeras ``size`` de cada una.

    Así ninguna fuente tapa al resto.

    Parameters
    ----------
    documents : dict[str, list[Document]]
        Documentos de cada fuente (ver ``load_documents``).
    size : int or None
        Máximo de instancias por fuente; ``None`` toma todas.
    window : WindowSpec or None, default=WindowSpec()
        Cómo cortar las ventanas; ``None`` usa una oración por instancia.

    Returns
    -------
    list[Sample]
        Las instancias de todas las fuentes, en el orden de ``documents``.
    """
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
    """Crea los datasets de puntuación y de capitalización a partir de las fuentes locales.

    Parameters
    ----------
    punctuation_size : int, default=PUNCTUATION_DATASET_SIZE
        Máximo de instancias por fuente para puntuación.
    capitalization_size : int, default=CAPITALIZATION_DATASET_SIZE
        Máximo de instancias por fuente para capitalización.
    window : WindowSpec or None, default=WindowSpec()
        Cómo cortar las ventanas; ``None`` usa una oración por instancia.

    Returns
    -------
    punctuation_dataset : PunctuationDataset
        El dataset completo para el problema de puntuación.
    capitalization_dataset : CapitalizationDataset
        El dataset completo para el problema de capitalización.
    """
    documents = load_documents()
    punctuation_dataset = PunctuationDataset(load_samples(documents, punctuation_size, window))
    capitalization_dataset = CapitalizationDataset(load_samples(documents, capitalization_size, window))

    return punctuation_dataset, capitalization_dataset


def split_dataset_dev_control(dataset: Dataset) -> tuple[Dataset, Dataset]:
    """Divide el dataset en desarrollo y control, con la semilla fija.

    Parameters
    ----------
    dataset : Dataset
        El dataset completo de un problema.

    Returns
    -------
    development_dataset : Dataset
        ``1 - CONTROL_RATIO`` de las instancias, para entrenar y elegir modelos.
    control_dataset : Dataset
        ``CONTROL_RATIO`` de las instancias, para la evaluación final.
    """

    total_size = len(dataset)
    control_size = int(CONTROL_RATIO * total_size)
    development_size = total_size - control_size

    development_dataset, control_dataset = random_split(
        dataset, [development_size, control_size], generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    return development_dataset, control_dataset


def split_train_test(development_dataset: Dataset) -> tuple[Dataset, Dataset]:
    """Divide el dataset de desarrollo en train y test, con la semilla fija.

    Con la misma semilla, todos los modelos se entrenan y evalúan sobre las mismas
    instancias.

    Parameters
    ----------
    development_dataset : Dataset
        El dataset de desarrollo (ver ``split_dataset_dev_control``).

    Returns
    -------
    train_dataset : Dataset
        ``1 - TEST_RATIO`` de las instancias.
    test_dataset : Dataset
        ``TEST_RATIO`` de las instancias.
    """
    train_dataset, test_dataset = random_split(
        development_dataset, [1 - TEST_RATIO, TEST_RATIO], generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    return train_dataset, test_dataset


def dataset_samples(dataset: Dataset) -> list[Sample]:
    """Devuelve las instancias de un dataset, atravesando los ``Subset`` de los splits.

    Parameters
    ----------
    dataset : Dataset
        Un ``TextDataset`` o un ``Subset`` (anidado o no) de uno.

    Returns
    -------
    list[Sample]
        Las palabras con su puntuación y capitalización originales, en el mismo
        orden que ``dataset[i]``.
    """
    if isinstance(dataset, Subset):
        samples = dataset_samples(dataset.dataset)
        return [samples[i] for i in dataset.indices]
    return dataset.samples


def build_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
    """Crea un DataLoader por batches de instancias.

    Parameters
    ----------
    dataset : Dataset
        Dataset de pares ``(tokens, etiquetas)``.
    shuffle : bool
        Si se mezclan las instancias en cada época.

    Returns
    -------
    DataLoader
        Batches de ``BATCH_SIZE`` instancias, con las etiquetas rellenadas con
        ``IGNORE_INDEX`` (ver ``collate_batch``).
    """
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, collate_fn=collate_batch)
