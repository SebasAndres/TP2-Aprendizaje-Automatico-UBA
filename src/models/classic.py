import numpy as np
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from src.features.base import TextFeatureExtractor

def train_classic_model(
    train_loader: DataLoader, 
    test_loader: DataLoader,
    feature_extractor: TextFeatureExtractor
):
    def tensor_a_numpy(loader: DataLoader):
        """Desempaqueta los lotes, extrae los atributos y los convierte a NumPy"""
        X_list, Y_list = [], []
        
        for tokens_batch, y_batch in loader:
            # Transformamos el texto crudo en la matriz de atributos numéricos
            x_features = feature_extractor.extract(tokens_batch)
        
            X_list.append(x_features.numpy())
            Y_list.append(y_batch.numpy())    
        return np.vstack(X_list), np.vstack(Y_list)

# Entrenamiento
    X_train, y_train = tensor_a_numpy(train_loader)
    modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    modelo.fit(X_train, y_train)

# Evaluacion

    X_test, y_test = tensor_a_numpy(test_loader)
    y_pred = modelo.predict(X_test)

    print("--- Resultados en Test ---")
    if y_test.ndim == 1 or y_test.shape[1] == 1:
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"F1 Macro (Capitalización): {f1:.4f}")
    else:
        f1_ini = f1_score(y_test[:, 0], y_pred[:, 0], average='macro')
        f1_fin = f1_score(y_test[:, 1], y_pred[:, 1], average='macro')
        print(f"F1 Macro (Puntuación Inicial): {f1_ini:.4f}")
        print(f"F1 Macro (Puntuación Final): {f1_fin:.4f}")
    return modelo