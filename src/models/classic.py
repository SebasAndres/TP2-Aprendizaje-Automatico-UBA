from torch.utils.data import DataLoader

from src.features.base import TextFeatureExtractor

def train_classic_model(
    train_loader: DataLoader, 
    test_loader: DataLoader,
    feature_extractor: TextFeatureExtractor
):
    breakpoint()
