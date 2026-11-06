import torch
from src.features.base import TextFeatureExtractor

VOCALES = set("aeiouáéíóú")
ACENTOS = set("áéíóú")


def _token_features(token: str, is_first: bool) -> list[float]:
    """Devuelve los atributos de un token según su texto y su posición en la oración"""
    # 1. ¿Es un sub-token de BERT? (empieza con ##)
    is_subtoken = 1.0 if token.startswith("##") else 0.0

    # Limpiamos el token para contar letras reales
    clean_token = token.replace("##", "")

    # 2. Longitud real del token
    length = float(len(clean_token))

    # 3. Cantidad de vocales
    num_vowels = float(sum(1 for char in clean_token if char in VOCALES))

    # 4. Cantidad de consonantes (longitud - vocales)
    num_consonants = length - num_vowels

    # 5. Proporción de vocales (para evitar que tokens largos dominen)
    vowel_ratio = num_vowels / length if length > 0 else 0.0

    # 6. Contamos cuántos acentos tiene el token
    num_acentos = float(sum(1 for char in clean_token if char in ACENTOS))

    # 7. ¿Es el primer token de la oración?
    is_first_token = 1.0 if is_first else 0.0

    return [is_subtoken, length, num_vowels, num_consonants, vowel_ratio, num_acentos, is_first_token]


class ClassicFeatureExtractor(TextFeatureExtractor):
    """Feature Extractor sin embeddings"""

    def extract(self, tokens: list[str]) -> torch.Tensor:
        features = [_token_features(token, i == 0) for i, token in enumerate(tokens)]
        # Devolvemos un tensor de dimensiones (n_tokens, n_features)
        return torch.tensor(features, dtype=torch.float32)
