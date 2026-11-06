import logging

from torch.utils.data import DataLoader

from src.features import TextFeatureExtractor
from src.problem import Problem

logger = logging.getLogger(__name__)

def train_standard_rnn_model(
    train_loader: DataLoader, 
    test_loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    problem: Problem
) -> tuple[any, dict[str, float]]:
    logger.warning("RNN estándar sin implementar: no hay modelo ni score")
    return None, {}