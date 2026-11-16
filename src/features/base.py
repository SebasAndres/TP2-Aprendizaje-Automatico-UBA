from abc import ABC, abstractmethod

import torch

from src.words import first_token_mask, token_word_ids


class TextFeatureExtractor(ABC):

    @abstractmethod
    def extract(self, tokens: list[str]) -> torch.Tensor:
        """Recibe la secuencia de tokens BERT (ej: ['cu', '##ánd', '##o']) y devuelve sus
        features por token: (n_tokens, n_features)"""
        pass

    def extract_words(self, tokens: list[str]) -> torch.Tensor:
        """Features por palabra, para los modelos clásicos: (n_palabras, n_features).
        Por defecto, las del primer token de cada palabra (en el promedio de embeddings
        todos los tokens de la palabra tienen la misma fila)"""
        return self.extract(tokens)[torch.from_numpy(first_token_mask(token_word_ids(tokens)))]
