import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.constants import BATCH_SIZE, SPLIT_SEED, TEST_RATIO
from src.data import collate_batch
from src.features import (
    AverageEmbeddingsExtractor,
    ClassicFeatureExtractor,
    EmbeddingsExtractor,
    TextFeatureExtractor,
)
from src.problem import Problem
from src.training.bidirectional_rnn import train_bidirectional_rnn_model
from src.training.classic import train_classic_model
from src.training.standard_rnn import train_standard_rnn_model

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Experiment:
    """Un modelo a entrenar junto con el extractor de features que recibe"""
    name: str
    trainer: Callable[..., tuple[Any, dict[str, float]]]
    feature_extractor: TextFeatureExtractor


def build_experiments() -> list[Experiment]:
    """Define las combinaciones de modelo y features que pide el enunciado"""
    embeddings = EmbeddingsExtractor()
    return [
        Experiment("classic_features", train_classic_model, ClassicFeatureExtractor()),
        Experiment("classic_embeddings", train_classic_model, AverageEmbeddingsExtractor()),
        Experiment("standard_rnn", train_standard_rnn_model, embeddings),
        Experiment("bidirectional_rnn", train_bidirectional_rnn_model, embeddings),
    ]


def _build_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_batch
    )

def train_models(
    development_dataset: Dataset,
    experiments: list[Experiment],
    problem: Problem
) -> dict[str, any]:
    """ 
    Acá se entrena y devuelve TODOS los modelos para el dataset dado.
    Inicialmente probamos los mismos modelos para ambos problemas. 
    """

    train_dataset, test_dataset = random_split(
        development_dataset,
        [1-TEST_RATIO, TEST_RATIO],
        generator=torch.Generator().manual_seed(SPLIT_SEED)
    )
    logger.info("Split: train=%d, test=%d", len(train_dataset), len(test_dataset))

    train_loader = _build_loader(train_dataset, shuffle=True)
    test_loader = _build_loader(test_dataset, shuffle=False)

    results = {}
    for experiment in experiments:
        logger.info("Entrenando modelo %s", experiment.name)
        model, score = experiment.trainer(train_loader, test_loader, experiment.feature_extractor, problem)
        logger.info("Modelo %s terminado: score=%s", experiment.name, score)
        results[experiment.name] = {"model": model, "score": score}
    return results

def save_models(models):
    return