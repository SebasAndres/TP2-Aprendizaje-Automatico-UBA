import torch

from src.constants import PUNCTUATION_MARKS
from src.data.base import TextDataset

Puntuacion = int

# Clases de puntuación. Las finales están ordenadas por prioridad: si una palabra termina
# con varios signos (ej: "...?") se queda con el de mayor valor.
NO_PUNCTUATION: Puntuacion = 0
OPEN_QUESTION: Puntuacion = 1   # ¿
COMMA: Puntuacion = 1           # ,
PERIOD: Puntuacion = 2          # .
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
    """Devuelve <PuntuacionInicial, PuntuacionFinal> según los signos en los bordes de la palabra"""
    prefix = word[:len(word) - len(word.lstrip(PUNCTUATION_MARKS))]
    suffix = word[len(word.rstrip(PUNCTUATION_MARKS)):]
    initial = OPEN_QUESTION if "¿" in prefix else NO_PUNCTUATION
    final = max((FINAL_MARKS.get(char, NO_PUNCTUATION) for char in suffix), default=NO_PUNCTUATION)
    return initial, final


class PunctuationDataset(TextDataset):

    def _expected_output(self, words: list[str], word_tokens: list[list[str]]) -> torch.Tensor:
        """
        Devuelve un tensor (n_tokens, 2) con <PuntuacionInicial, PuntuacionFinal> para cada token.
        La inicial va al primer token de la palabra y la final al último. Los signos sueltos
        (palabras sin tokens, ej: "¿ qué ?") se asignan al token siguiente o anterior.
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
