"""Utilidades para pasar de tokens BERT a palabras: índices, máscaras y voto por palabra."""

from collections import Counter

import numpy as np


def token_word_ids(tokens: list[str]) -> np.ndarray:
    """Asigna a cada token el índice de la palabra a la que pertenece.

    Un token con prefijo ``"##"`` continúa la palabra anterior. El primer token
    siempre inicia una palabra, aunque tenga el prefijo.

    Parameters
    ----------
    tokens : list[str]
        Tokens BERT de una instancia.

    Returns
    -------
    np.ndarray of shape (n_tokens,)
        Índice de palabra de cada token, empezando en 0.

    Examples
    --------
    >>> token_word_ids(["hola", "mun", "##do"])
    array([0, 1, 1])
    """
    return np.cumsum([i == 0 or not token.startswith("##") for i, token in enumerate(tokens)]) - 1


def split_words(tokens: list[str]) -> list[str]:
    """Junta los tokens de cada palabra, con el mismo criterio que ``token_word_ids``.

    Parameters
    ----------
    tokens : list[str]
        Tokens BERT de una instancia.

    Returns
    -------
    list[str]
        Una palabra por cada grupo de tokens, sin los prefijos ``"##"``.

    Examples
    --------
    >>> split_words(["cu", "##ánd", "##o", "va"])
    ['cuándo', 'va']
    """
    words: list[str] = []
    for i, token in enumerate(tokens):
        if i > 0 and token.startswith("##"):
            words[-1] += token.removeprefix("##")
        else:
            words.append(token.removeprefix("##"))
    return words


def concat_word_ids(word_ids_list: list[np.ndarray]) -> np.ndarray:
    """Concatena los índices de palabra de varias instancias sin que se repitan entre ellas.

    Parameters
    ----------
    word_ids_list : list[np.ndarray]
        Índices de palabra de cada instancia (ver ``token_word_ids``). Cada arreglo
        debe empezar en 0 y solo el último puede estar vacío.

    Returns
    -------
    np.ndarray
        Todos los índices concatenados, desplazados para que cada instancia
        continúe la numeración de la anterior.

    Examples
    --------
    >>> concat_word_ids([np.array([0, 0, 1]), np.array([0, 1])])
    array([0, 0, 1, 2, 3])
    """
    offsets = np.cumsum([0] + [word_ids[-1] + 1 for word_ids in word_ids_list[:-1]])
    return np.concatenate([word_ids + offset for word_ids, offset in zip(word_ids_list, offsets)])


def first_token_mask(word_ids: np.ndarray) -> np.ndarray:
    """Marca el primer token de cada palabra.

    Parameters
    ----------
    word_ids : np.ndarray of shape (n_tokens,)
        Índice de palabra de cada token.

    Returns
    -------
    np.ndarray of shape (n_tokens,)
        Máscara booleana, ``True`` en el primer token de cada palabra.

    Examples
    --------
    >>> first_token_mask(np.array([0, 0, 1]))
    array([ True, False,  True])
    """
    return np.r_[True, word_ids[1:] != word_ids[:-1]]


def last_token_mask(word_ids: np.ndarray) -> np.ndarray:
    """Marca el último token de cada palabra.

    Parameters
    ----------
    word_ids : np.ndarray of shape (n_tokens,)
        Índice de palabra de cada token.

    Returns
    -------
    np.ndarray of shape (n_tokens,)
        Máscara booleana, ``True`` en el último token de cada palabra.

    Examples
    --------
    >>> last_token_mask(np.array([0, 0, 1]))
    array([False,  True,  True])
    """
    return np.r_[word_ids[1:] != word_ids[:-1], True]


def majority_by_word(pred: np.ndarray, word_ids: np.ndarray) -> np.ndarray:
    """Reduce predicciones por token a una por palabra, tomando la clase más votada.

    Un empate lo define la clase que aparece primero en la palabra.

    Parameters
    ----------
    pred : np.ndarray of shape (n_tokens,)
        Clase predicha de cada token.
    word_ids : np.ndarray of shape (n_tokens,)
        Índice de palabra de cada token.

    Returns
    -------
    np.ndarray of shape (n_palabras,)
        Clase más votada de cada palabra.

    Examples
    --------
    >>> majority_by_word(np.array([2, 1, 1, 3, 4]), np.array([0, 0, 0, 1, 1]))
    array([1, 3])
    """
    word_starts = np.flatnonzero(first_token_mask(word_ids))
    words = np.split(pred, word_starts[1:])
    return np.array([Counter(word.tolist()).most_common(1)[0][0] for word in words])
