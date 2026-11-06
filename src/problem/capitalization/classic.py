from sklearn.ensemble import RandomForestClassifier


def create_classic_model() -> RandomForestClassifier:
    """Crea el modelo clásico de capitalización, sin entrenar"""
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
