"""Diccionario de capitalización: cuántas veces apareció cada palabra con cada capitalización.

Es la sugerencia del enunciado para casos como "onu" -> MAYÚS o "mcdonald's" -> MiXta, que
no se pueden deducir de las letras de la palabra. Se arma solo con instancias de train.

Las palabras son las mismas que ven los modelos clásicos: los tokens unidos por "##" (ver
split_words), así "mcdonald's" son 3 palabras: "mcdonald", "'" y "s".
"""

import numpy as np

from src.data import CapitalizationDataset
from src.data.capitalization_dataset import N_CAPITALIZATION_CLASSES
from src.data.samples import Sample
from src.words import first_token_mask, split_words, token_word_ids

# Atributos por palabra: vista, log(1 + veces vista) y la proporción de cada capitalización
N_DICTIONARY_FEATURES = 2 + N_CAPITALIZATION_CLASSES


class CapitalizationDictionary:
    """Para cada palabra vista en `samples`, cuántas veces apareció con cada capitalización"""

    def __init__(self, samples: list[Sample]):
        self.counts: dict[str, np.ndarray] = {}
        for tokens, labels in CapitalizationDataset(samples):
            if not tokens:
                continue
            # todos los tokens de una palabra tienen su capitalización: tomamos la del primero
            classes = labels.numpy()[first_token_mask(token_word_ids(tokens))]
            for word, capitalization in zip(split_words(tokens), classes):
                self.counts.setdefault(word, np.zeros(N_CAPITALIZATION_CLASSES))[capitalization] += 1

    def features(self, words: list[str]) -> np.ndarray:
        """(n_palabras, N_DICTIONARY_FEATURES): [vista, log(1 + veces vista), proporción de
        minúscula, Inicial, MiXta y MAYÚS]. Una palabra que no se vio tiene todo en cero"""
        rows = np.zeros((len(words), N_DICTIONARY_FEATURES), dtype=np.float32)
        for i, word in enumerate(words):
            counts = self.counts.get(word)
            if counts is not None:
                total = counts.sum()
                rows[i] = [1.0, np.log1p(total), *(counts / total)]
        return rows
