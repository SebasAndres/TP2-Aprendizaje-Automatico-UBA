from sklearn.ensemble import RandomForestClassifier

from src.constants import RANDOM_SEED

def create_classic_model() -> RandomForestClassifier:
    """Crea el modelo clásico de capitalización, sin entrenar"""
    return RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
