from src.features.base import TextFeatureExtractor
from src.features.classic import ClassicFeatureExtractor
from src.features.embeddings import EmbeddingsExtractor, AverageEmbeddingsExtractor

__all__ = [
    "TextFeatureExtractor",
    "ClassicFeatureExtractor",
    "EmbeddingsExtractor",
    "AverageEmbeddingsExtractor",
]
