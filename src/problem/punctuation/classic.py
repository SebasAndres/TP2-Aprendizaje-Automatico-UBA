from sklearn.ensemble import RandomForestClassifier

from src.constants import RANDOM_SEED


def create_classic_model() -> RandomForestClassifier:
    """Crea el modelo clásico de puntuación, sin entrenar.

    Returns
    -------
    RandomForestClassifier
        Random forest de 100 árboles con la semilla fija.
    """
    return RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
