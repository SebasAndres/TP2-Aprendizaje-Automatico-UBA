from torch.utils.data import Dataset, DataLoader, random_split
from src.constants import BATCH_SIZE

def train_models(development_dataset: Dataset) -> dict[str, any]:
    """ 
    Acá se entrena y devuelve TODOS los modelos para el dataset dado.
    Inicialmente probamos los mismos modelos para ambos problemas. 
    """

    train_dataset, test_dataset = random_split(development_dataset, [])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  
        num_workers=0
    )

    # Acá para cada modelo posible entrenar (!)

    return {
        "modelo-1": None,
        "...": None
    }

def save_models(models):
    return