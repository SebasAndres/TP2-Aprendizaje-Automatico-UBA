import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from joint_rnn import JointRNN
from src.constants import BATCH_SIZE, RANDOM_SEED, TEST_RATIO
from src.data import collate_batch
from src.features import (
    AverageEmbeddingsExtractor,
    ClassicFeatureExtractor,
    EmbeddingsExtractor,
    TextFeatureExtractor,
)
from src.problem import Problem
from src.training.classic import train_classic_model
from src.training.neural import neural_trainer

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
        Experiment("standard_rnn", neural_trainer(lambda problem: problem.standard_rnn_model()), embeddings),
        Experiment("bidirectional_rnn", neural_trainer(lambda problem: problem.bidirectional_rnn_model()), embeddings),
        Experiment("joint_rnn", neural_trainer(lambda _: JointRNN()), embeddings),
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
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    logger.info("Split: train=%d, test=%d", len(train_dataset), len(test_dataset))

    train_loader = _build_loader(train_dataset, shuffle=True)
    test_loader = _build_loader(test_dataset, shuffle=False)

    results = {}
    for experiment in experiments:
        logger.info("Entrenando modelo %s", experiment.name)
        model, score = experiment.trainer(train_loader, test_loader, experiment.feature_extractor, problem)

        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), f"/models/{problem.__class__.__name__}/{experiment.name}.pkl")
        
        logger.info("Modelo %s terminado: score=%s", experiment.name, score)
        results[experiment.name] = {"model": model, "score": score}
    return results