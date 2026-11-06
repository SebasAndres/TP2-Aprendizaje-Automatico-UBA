from collections import Counter

import numpy as np


def token_word_ids(tokens: list[str]) -> np.ndarray:
    """Asigna a cada token el índice de la palabra a la que pertenece.

    Un token con prefijo "##" continúa la palabra anterior. El primer token siempre
    inicia una palabra, aunque tenga el prefijo.

    Ejemplo: ["hola", "mun", "##do"] -> [0, 1, 1]
    """
    return np.cumsum([i == 0 or not token.startswith("##") for i, token in enumerate(tokens)]) - 1


def concat_word_ids(word_ids_list: list[np.ndarray]) -> np.ndarray:
    """Concatena los índices de palabra de varias instancias sin que se repitan entre ellas.

    Cada arreglo debe empezar en 0 y solo el último puede estar vacío.

    Ejemplo: [[0, 0, 1], [0, 1]] -> [0, 0, 1, 2, 3]
    """
    offsets = np.cumsum([0] + [word_ids[-1] + 1 for word_ids in word_ids_list[:-1]])
    return np.concatenate([word_ids + offset for word_ids, offset in zip(word_ids_list, offsets)])


def first_token_mask(word_ids: np.ndarray) -> np.ndarray:
    """Devuelve una máscara booleana que marca el primer token de cada palabra.

    Ejemplo: [0, 0, 1] -> [True, False, True]
    """
    return np.r_[True, word_ids[1:] != word_ids[:-1]]


def last_token_mask(word_ids: np.ndarray) -> np.ndarray:
    """Devuelve una máscara booleana que marca el último token de cada palabra.

    Ejemplo: [0, 0, 1] -> [False, True, True]
    """
    return np.r_[word_ids[1:] != word_ids[:-1], True]


def majority_by_word(pred: np.ndarray, word_ids: np.ndarray) -> np.ndarray:
    """Reduce predicciones por token a una por palabra, tomando la clase más votada.

    Un empate lo define la clase que aparece primero en la palabra.

    Ejemplo: pred=[2, 1, 1, 3, 4], word_ids=[0, 0, 0, 1, 1] -> [1, 3]
    """
    word_starts = np.flatnonzero(first_token_mask(word_ids))
    words = np.split(pred, word_starts[1:])
    return np.array([Counter(word.tolist()).most_common(1)[0][0] for word in words])
