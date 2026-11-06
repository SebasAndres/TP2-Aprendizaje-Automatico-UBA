from collections import Counter

import numpy as np


def token_word_ids(tokens: list[str]) -> np.ndarray:
    """Asigna a cada token el índice de su palabra: los sub-tokens (##) continúan la anterior"""
    return np.cumsum([i == 0 or not token.startswith("##") for i, token in enumerate(tokens)]) - 1


def concat_word_ids(word_ids_list: list[np.ndarray]) -> np.ndarray:
    """Une los índices de varias instancias para que ninguna palabra comparta índice"""
    offsets = np.cumsum([0] + [word_ids[-1] + 1 for word_ids in word_ids_list[:-1]])
    return np.concatenate([word_ids + offset for word_ids, offset in zip(word_ids_list, offsets)])


def first_token_mask(word_ids: np.ndarray) -> np.ndarray:
    """Marca el primer token de cada palabra"""
    return np.r_[True, word_ids[1:] != word_ids[:-1]]


def last_token_mask(word_ids: np.ndarray) -> np.ndarray:
    """Marca el último token de cada palabra"""
    return np.r_[word_ids[1:] != word_ids[:-1], True]


def majority_by_word(pred: np.ndarray, word_ids: np.ndarray) -> np.ndarray:
    """Devuelve la clase más votada por los tokens de cada palabra. Un empate lo define
    la clase que aparece primero en la palabra"""
    word_starts = np.flatnonzero(first_token_mask(word_ids))
    words = np.split(pred, word_starts[1:])
    return np.array([Counter(word.tolist()).most_common(1)[0][0] for word in words])
