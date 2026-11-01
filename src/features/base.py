from abc import ABC, abstractmethod


class TextFeatureExtractor(ABC):

    @abstractmethod
    def extract(self, tokens: list[str]):
        """Recibe la secuencia de tokens BERT (ej: ['cu', '##ánd', '##o']) y devuelve sus features"""
        pass