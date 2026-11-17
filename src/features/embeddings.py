import torch
from transformers import BertModel, BertTokenizer
from src.constants import EMBEDDINGS_MODEL

from src.features.base import TextFeatureExtractor
from src.words import token_word_ids


class EmbeddingsExtractor(TextFeatureExtractor):
    """Feature extractor con los embeddings estáticos de BERT, token a token.

    Parameters
    ----------
    embeddings_model : str, default=EMBEDDINGS_MODEL
        Nombre del modelo de ``transformers`` del que se toman el tokenizador y los
        embeddings.

    Attributes
    ----------
    tokenizer : BertTokenizer
        Tokenizador, para pasar de tokens a ids.
    embeddings : BertModel
        Modelo BERT en modo eval; solo se usa su tabla de embeddings.
    """

    def __init__(self, embeddings_model: str = EMBEDDINGS_MODEL) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(embeddings_model)
        self.embeddings = BertModel.from_pretrained(embeddings_model)
        self.embeddings.eval()

    def extract(self, tokens: list[str]) -> torch.Tensor:
        """Busca el embedding de cada token.

        Los tokens fuera del vocabulario usan el embedding de [UNK], porque hay que
        predecir sobre todos los tokens.

        Parameters
        ----------
        tokens : list[str]
            Tokens BERT de una instancia.

        Returns
        -------
        torch.Tensor of shape (n_tokens, embedding_dim)
            El embedding de cada token.
        """

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        with torch.no_grad():
            word_embeddings = self.embeddings.embeddings.word_embeddings
            return word_embeddings(torch.tensor(token_ids))


class AverageEmbeddingsExtractor(EmbeddingsExtractor):
    """Feature extractor que representa cada token con el promedio de embeddings de su palabra."""

    def extract(self, tokens: list[str]) -> torch.Tensor:
        """Promedia los embeddings de los tokens de cada palabra.

        Parameters
        ----------
        tokens : list[str]
            Tokens BERT de una instancia.

        Returns
        -------
        torch.Tensor of shape (n_tokens, embedding_dim)
            Los tokens de una misma palabra comparten fila.
        """
        embeddings = super().extract(tokens)
        word_ids = torch.from_numpy(token_word_ids(tokens))
        n_words = int(word_ids[-1]) + 1
        totals = torch.zeros(n_words, embeddings.size(1)).index_add_(0, word_ids, embeddings)
        averages = totals / torch.bincount(word_ids).unsqueeze(1)
        return averages[word_ids]
