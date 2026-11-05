import torch
from src.features.base import TextFeatureExtractor

class ClassicFeatureExtractor(TextFeatureExtractor):
    """Feature Extractor sin embeddings"""

    def extract(self, tokens: list[str]) -> torch.Tensor:
        features = []
        vocales = set("aeiouáéíóú")
        
        for token in tokens:
            # 1. ¿Es un sub-token de BERT? (empieza con ##)
            is_subtoken = 1.0 if token.startswith("##") else 0.0
            
            # Limpiamos el token para contar letras reales
            clean_token = token.replace("##", "")
            
            # 2. Longitud real del token
            length = float(len(clean_token))
            
            # 3. Cantidad de vocales
            num_vowels = float(sum(1 for char in clean_token if char in vocales))
            
            # 4. Cantidad de consonantes (longitud - vocales)
            num_consonants = length - num_vowels
            
            # 5. Proporción de vocales (para evitar que tokens largos dominen)
            vowel_ratio = num_vowels / length if length > 0 else 0.0
            
            # Agregamos los atributos de este token como una lista
            token_features = [is_subtoken, length, num_vowels, num_consonants, vowel_ratio]
            features.append(token_features)
            
        # Devolvemos un tensor de dimensiones (n_tokens, n_features)
        return torch.tensor(features, dtype=torch.float32)