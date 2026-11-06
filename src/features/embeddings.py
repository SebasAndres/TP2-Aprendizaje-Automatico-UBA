import torch
from transformers import BertModel, BertTokenizer
from src.constants import EMBEDDINGS_MODEL

from src.features.base import TextFeatureExtractor
from src.words import token_word_ids

class EmbeddingsExtractor(TextFeatureExtractor):
    """Feature Extractor con embeddings, es token-a-token"""

    def __init__(self, embeddings_model: str = EMBEDDINGS_MODEL):
        self.tokenizer = BertTokenizer.from_pretrained(embeddings_model)
        self.embeddings = BertModel.from_pretrained(embeddings_model)
        self.embeddings.eval()

    def extract(self, tokens: list[str]) -> torch.Tensor:
        """Devuelve un tensor (n_tokens, embedding_dim). Los tokens fuera del vocabulario usan
        el embedding de [UNK], porque hay que predecir sobre todos los tokens"""

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        with torch.no_grad():
            word_embeddings = self.embeddings.embeddings.word_embeddings
            return word_embeddings(torch.tensor(token_ids))


class AverageEmbeddingsExtractor(EmbeddingsExtractor):
    """Feature Extractor que representa cada token con el promedio de embeddings de su palabra"""

    def extract(self, tokens: list[str]) -> torch.Tensor:
        """Devuelve un tensor (n_tokens, embedding_dim): los tokens de una palabra comparten fila"""
        embeddings = super().extract(tokens)
        word_ids = torch.from_numpy(token_word_ids(tokens))
        n_words = int(word_ids[-1]) + 1
        totals = torch.zeros(n_words, embeddings.size(1)).index_add_(0, word_ids, embeddings)
        averages = totals / torch.bincount(word_ids).unsqueeze(1)
        return averages[word_ids]
