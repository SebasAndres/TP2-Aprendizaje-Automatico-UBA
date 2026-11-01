import torch
from src.features.base import TextFeatureExtractor

class ClassicFeatureExtractor(TextFeatureExtractor):
    """Feature Extractor sin embeddings"""

    def extract(self, tokens: list[str]) -> torch.Tensor:
        breakpoint()
        return torch.tensor([0,1,1,0])