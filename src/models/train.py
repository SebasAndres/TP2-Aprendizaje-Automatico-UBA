from torch.utils.data import Dataset, DataLoader, random_split

from src.constants import BATCH_SIZE, TEST_RATIO
from src.models.classic import train_classic_model
from src.models.standard_rnn import train_standard_rnn_model
from src.models.bidirectional_rnn import train_bidirectional_rnn_model
from src.features.base import TextFeatureExtractor

def train_models(
    development_dataset: Dataset,
    feature_extractor: TextFeatureExtractor
) -> dict[str, any]:
    """ 
    Acá se entrena y devuelve TODOS los modelos para el dataset dado.
    Inicialmente probamos los mismos modelos para ambos problemas. 
    """

    train_dataset, test_dataset = random_split(development_dataset, [1-TEST_RATIO, TEST_RATIO])

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
    classic_model = train_classic_model(train_loader, test_loader, feature_extractor)
    standard_rnn_model = train_standard_rnn_model(train_loader, test_loader, feature_extractor)
    bidirectional_rnn_model = train_bidirectional_rnn_model(train_loader, test_loader, feature_extractor)

    return {
        "classic_model": classic_model,
        "standard_rnn_model": standard_rnn_model,
        "bidirectional_rnn_model": bidirectional_rnn_model
    }

def save_models(models):
    return