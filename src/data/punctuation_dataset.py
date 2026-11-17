import torch

from src.constants import PUNCTUATION_MARKS
from src.data.base import TextDataset

Puntuacion = int

# Clases de puntuación. Las finales están ordenadas por prioridad: si una palabra termina
# con varios signos (ej: "...?") se queda con el de mayor valor.
NO_PUNCTUATION: Puntuacion = 0
OPEN_QUESTION: Puntuacion = 1  # ¿
COMMA: Puntuacion = 1  # ,
PERIOD: Puntuacion = 2  # .
CLOSE_QUESTION: Puntuacion = 3  # ?

INITIAL_LABELS = ["", "¿"]
FINAL_LABELS = ["", ",", ".", "?"]

FINAL_MARKS: dict[str, Puntuacion] = {
    ",": COMMA,
    ".": PERIOD,
    "!": PERIOD,  # no es una clase: se trata como fin de oración
    "?": CLOSE_QUESTION,
}


def word_punctuation(word: str) -> tuple[Puntuacion, Puntuacion]:
    """Clasifica la puntuación inicial y final según los signos en los bordes de la palabra.

    Parameters
    ----------
    word : str
        Palabra con su puntuación original.

    Returns
    -------
    initial : Puntuacion
        ``OPEN_QUESTION`` si la palabra empieza con "¿", si no ``NO_PUNCTUATION``.
    final : Puntuacion
        El signo final de mayor prioridad (ver ``FINAL_MARKS``), o ``NO_PUNCTUATION``.

    Examples
    --------
    >>> word_punctuation("¿Qué?")
    (1, 3)
    """
    prefix = word[: len(word) - len(word.lstrip(PUNCTUATION_MARKS))]
    suffix = word[len(word.rstrip(PUNCTUATION_MARKS)) :]
    initial = OPEN_QUESTION if "¿" in prefix else NO_PUNCTUATION
    final = max((FINAL_MARKS.get(char, NO_PUNCTUATION) for char in suffix), default=NO_PUNCTUATION)
    return initial, final


class PunctuationDataset(TextDataset):
    """Dataset del problema de puntuación: ``[inicial, final]`` por token (ver ``TextDataset``)."""

    def _expected_output(self, words: list[str], word_tokens: list[list[str]]) -> torch.Tensor:
        """Asigna la puntuación inicial al primer token de cada palabra y la final al último.

        Los signos sueltos (palabras sin tokens, ej: "¿ qué ?") se asignan al token
        siguiente (la inicial) o al anterior (la final).

        Parameters
        ----------
        words : list[str]
            Palabras originales de la instancia.
        word_tokens : list[list[str]]
            Tokens de cada palabra normalizada.

        Returns
        -------
        torch.Tensor of shape (n_tokens, 2)
            Cada fila es ``[inicial, final]``, índices de ``INITIAL_LABELS`` y
            ``FINAL_LABELS``: inicial 0 = "", 1 = "¿"; final 0 = "", 1 = ",", 2 = ".", 3 = "?".

        Notes
        -----
        Para reconstruir: ``INITIAL_LABELS[y[i][0]] + token + FINAL_LABELS[y[i][1]]``.

        Examples
        --------
        "¿Cuándo vamos a McDonald's? Ellos no vienen hoy."::

            token    inicial  final
            cu          1       0     <- "¿" va al primer token de la palabra
            ##ánd       0       0
            ##o         0       0
            va          0       0
            ##mos       0       0
            a           0       0
            m           0       0
            ##c         0       0
            ##dona      0       0
            ##ld        0       0
            '           0       0
            s           0       3     <- "?" va al último token de la palabra
            ellos       0       0
            no          0       0
            viene       0       0
            ##n         0       0
            hoy         0       2     <- "."

        Un token puede tener las dos: "¿Qué?" -> qué -> [[1, 3]]

        Un signo suelto se pega al token anterior: "Hola , la" -> hol, ##a, la ->
        [[0, 0], [0, 1], [0, 0]]
        """
        labels: list[list[Puntuacion]] = []
        pending_initial = NO_PUNCTUATION
        for word, tokens in zip(words, word_tokens):
            initial, final = word_punctuation(word)
            if not tokens:
                pending_initial = max(pending_initial, initial)
                if labels:
                    labels[-1][1] = max(labels[-1][1], final)
                continue
            word_labels = [[NO_PUNCTUATION, NO_PUNCTUATION] for _ in tokens]
            word_labels[0][0] = max(initial, pending_initial)
            word_labels[-1][1] = final
            labels.extend(word_labels)
            pending_initial = NO_PUNCTUATION
        return torch.tensor(labels, dtype=torch.long).reshape(-1, 2)
