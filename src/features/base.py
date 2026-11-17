from abc import ABC, abstractmethod

import torch

from src.words import first_token_mask, token_word_ids


class TextFeatureExtractor(ABC):
    """Convierte los tokens BERT de una instancia en features numéricas.

    Las redes usan las features por token (``extract``) y los modelos clásicos las
    features por palabra (``extract_words``).
    """

    @abstractmethod
    def extract(self, tokens: list[str]) -> torch.Tensor:
        """Calcula las features de cada token.

        Parameters
        ----------
        tokens : list[str]
            Tokens BERT de una instancia, ej: ``['cu', '##ánd', '##o']``.

        Returns
        -------
        torch.Tensor of shape (n_tokens, n_features)
            Las features de cada token.
        """
        pass

    def extract_words(self, tokens: list[str]) -> torch.Tensor:
        """Calcula las features de cada palabra, para los modelos clásicos.

        Por defecto son las del primer token de cada palabra (en el promedio de embeddings
        todos los tokens de la palabra tienen la misma fila).

        Parameters
        ----------
        tokens : list[str]
            Tokens BERT de una instancia.

        Returns
        -------
        torch.Tensor of shape (n_palabras, n_features)
            Las features de cada palabra.
        """
        return self.extract(tokens)[torch.from_numpy(first_token_mask(token_word_ids(tokens)))]
