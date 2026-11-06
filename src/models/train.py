import logging

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.constants import BATCH_SIZE, SPLIT_SEED, TEST_RATIO
from src.data import collate_batch
from src.models import (
    train_classic_model,
    train_standard_rnn_model,
    train_bidirectional_rnn_model
)
from src.problem import Problem
from src.features.base import TextFeatureExtractor

logger = logging.getLogger(__name__)

TRAINERS = {
    "classic": train_classic_model,
    "standard_rnn": train_standard_rnn_model,
    "bidirectional_rnn": train_bidirectional_rnn_model
}

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
    feature_extractor: TextFeatureExtractor,
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
    for name, trainer in TRAINERS.items():
        logger.info("Entrenando modelo %s", name)
        model, score = trainer(train_loader, test_loader, feature_extractor, problem)
        logger.info("Modelo %s terminado: score=%s", name, score)
        results[name] = {"model": model, "score": score}
    return results

def save_models(models):
    return