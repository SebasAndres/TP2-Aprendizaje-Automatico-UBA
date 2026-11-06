from torch.utils.data import DataLoader

from src.features import TextFeatureExtractor
from src.metrics import Metric

def train_standard_rnn_model(
    train_loader: DataLoader, 
    test_loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    metric: Metric
):
    ...