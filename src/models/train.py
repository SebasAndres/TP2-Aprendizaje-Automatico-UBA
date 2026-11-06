from torch.utils.data import Dataset, DataLoader, random_split

from src.constants import BATCH_SIZE, TEST_RATIO
from src.data import collate_batch
from src.models import (
    train_classic_model,
    train_standard_rnn_model,
    train_bidirectional_rnn_model
)
from src.problem import Problem
from src.features.base import TextFeatureExtractor

def train_models(
    development_dataset: Dataset,
    feature_extractor: TextFeatureExtractor,
    problem: Problem
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
        num_workers=0,
        collate_fn=collate_batch
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  
        num_workers=0,
        collate_fn=collate_batch
    )

    (
        classic_model,
        classic_score
    ) = train_classic_model(
        train_loader, 
        test_loader, 
        feature_extractor, 
        problem
    )
    (
        standard_rnn_model,
        standard_rnn_score
    ) = train_standard_rnn_model(
        train_loader, 
        test_loader, 
        feature_extractor, 
        problem
    )
    (
        bidirectional_rnn_model,
        bidirectional_rnn_score
    ) = train_bidirectional_rnn_model(
        train_loader, 
        test_loader, 
        feature_extractor, 
        problem
    )

    return {
        "classic": {
            "model": classic_model,
            "score": classic_score
        },
        "standard_rnn": {
            "model": standard_rnn_model,
            "score": standard_rnn_score,
        },
        "bidirectional_rnn": {
            "model": bidirectional_rnn_model,
            "score": bidirectional_rnn_score
        }
    }

def save_models(models):
    return