import torch

from src.data.base import TextDataset, strip_punctuation

Capitalization = int

LOWERCASE: Capitalization = 0  # "hola"
TITLE: Capitalization = 1      # "Hola" (incluye palabras de 1 letra)
MIXED: Capitalization = 2      # "McDonald's", "iPhone"
UPPERCASE: Capitalization = 3  # "ONU", "UBA" (más de una letra)


def word_capitalization(word: str) -> Capitalization:
    """Clasifica la capitalización de una palabra considerando solo sus letras"""
    letters = [char for char in word if char.isalpha()]
    if not any(char.isupper() for char in letters):
        return LOWERCASE
    if len(letters) > 1 and all(char.isupper() for char in letters):
        return UPPERCASE
    if letters[0].isupper() and not any(char.isupper() for char in letters[1:]):
        return TITLE
    return MIXED


class CapitalizationDataset(TextDataset):

    def _expected_output(self, words: list[str], word_tokens: list[list[str]]) -> torch.Tensor:
        """Asigna la capitalización de cada palabra a todos sus tokens.

        Devuelve un tensor (n_tokens,) con un valor por token:
            0 = LOWERCASE, 1 = TITLE, 2 = MIXED, 3 = UPPERCASE

        Ejemplo: "¿Cuándo vamos a McDonald's? Ellos"
            tokens: cu  ##ánd  ##o  va  ##mos  a  m  ##c  ##dona  ##ld  '  s  ellos
            y:      1   1      1    0   0      0  2  2    2       2     2  2  1

        Ejemplo: "Hola, la UBA" -> hol, ##a, la, u, ##ba -> [1, 1, 0, 3, 3]

        Para reconstruir una palabra hay que juntar sus tokens (los "##" continúan la anterior)
        y aplicar la clase de la palabra: 1 -> capitalize(), 3 -> upper(). La clase 2 no dice
        qué letras van en mayúscula (ej: "McDonald's"), así que no se puede reconstruir solo con y.
        """
        labels = [
            word_capitalization(strip_punctuation(word))
            for word, tokens in zip(words, word_tokens)
            for _ in tokens
        ]
        return torch.tensor(labels, dtype=torch.long)
