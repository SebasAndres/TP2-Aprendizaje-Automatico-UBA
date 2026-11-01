import torch
from transformers import BertModel, BertTokenizer
from src.constants import EMBEDDINGS_MODEL

from src.features.base import TextFeatureExtractor

class EmbeddingsExtractor(TextFeatureExtractor):
    """Feature Extractor con embeddings, es token-a-token"""

    def __init__(self, embeddings_model: str = EMBEDDINGS_MODEL):
        self.tokenizer = BertTokenizer.from_pretrained(embeddings_model)
        self.embeddings = BertModel.from_pretrained(embeddings_model)
        self.embeddings.eval()

    def extract(self, tokens: list[str]) -> torch.Tensor | None:
        """Convierte texto a su representación tensorial usando embeddings """

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if any(token_id is None or token_id == self.tokenizer.unk_token_id for token_id in token_ids):
            return None
        with torch.no_grad():
            word_embeddings = self.embeddings.embeddings.word_embeddings
            return word_embeddings(torch.tensor(token_ids))


class AverageEmbeddingsExtractor(EmbeddingsExtractor):
    """
    Feature Extractor con embeddings, token-a-token con contexto de la oración: concatena
    a cada token el promedio de los embeddings de todos los tokens de la oración
    """

    def extract(self, tokens: list[str]) -> torch.Tensor | None:
        """
        Devuelve un tensor (n_tokens, 2 * embedding_dim) donde cada fila es
        [embedding del token, promedio de embeddings de la oración]
        """

        tokens_embeddings = super().extract(tokens)
        if tokens_embeddings is None:
            return None
        avg_embedding = tokens_embeddings.mean(dim=0, keepdim=True).expand_as(tokens_embeddings)
        return torch.cat([tokens_embeddings, avg_embedding], dim=1)