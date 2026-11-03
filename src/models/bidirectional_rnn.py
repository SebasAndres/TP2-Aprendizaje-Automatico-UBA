from torch.utils.data import DataLoader

from features.base import TextFeatureExtractor

def train_bidirectional_rnn_model(
    train_loader: DataLoader, 
    test_loader: DataLoader,
    feature_extractor: TextFeatureExtractor
):
    ...