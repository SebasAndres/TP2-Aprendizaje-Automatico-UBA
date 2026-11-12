import logging
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.features import TextFeatureExtractor
from src.problem import Problem
from src.words import concat_word_ids, token_word_ids

logger = logging.getLogger(__name__)

N_EPOCHS = 5
LEARNING_RATE = 1e-3
LOG_EVERY_N_BATCHES = 50


def _batch_to_tensors(
    batch_X: list[list[str]],
    batch_y: torch.Tensor,
    feature_extractor: TextFeatureExtractor
) -> tuple[torch.Tensor, torch.Tensor, list[list[str]]] | None:
    keep = [i for i, tokens in enumerate(batch_X) if tokens]
    if not keep:
        return None
    tokens = [batch_X[i] for i in keep]
    X = pad_sequence([feature_extractor.extract(sentence) for sentence in tokens], batch_first=True)
    Y = batch_y[keep, :X.size(1)]
    return X, Y, tokens


def _predict(
    model: torch.nn.Module,
    dataloader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, reals, word_ids = [], [], []
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch = _batch_to_tensors(batch_X, batch_y, feature_extractor)
            if batch is None:
                continue
            X, Y, tokens = batch
            pred = model(X.to(device)).argmax(dim=-1).cpu()
            for i, sentence in enumerate(tokens):
                n_tokens = len(sentence)
                preds.append(pred[i, :n_tokens].numpy())
                reals.append(Y[i, :n_tokens].numpy())
                word_ids.append(token_word_ids(sentence))
    return np.concatenate(preds), np.concatenate(reals), concat_word_ids(word_ids)


def train_bidirectional_rnn_model(
    train_loader: DataLoader, 
    test_loader: DataLoader,
    feature_extractor: TextFeatureExtractor,
    problem: Problem
) -> tuple[any, dict[str, float]]:
    
    try:
        # AQUÍ CAMBIA: Instanciamos el modelo bidireccional
        model = problem.bidirectional_rnn_model()
    except NotImplementedError as error:
        logger.warning("RNN bidireccional sin implementar para %s: %s", type(problem).__name__, error)
        return None, {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Entrenando %s en %s (%d parámetros)", type(model).__name__, device, n_params)

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for i, (batch_X, batch_y) in enumerate(train_loader, start=1):
            batch = _batch_to_tensors(batch_X, batch_y, feature_extractor)
            if batch is None:
                continue
            X, Y, _ = batch

            logits = model(X.to(device))                     
            loss = problem.metric.loss(logits, Y.to(device))  
            optimizer.zero_grad()  
            loss.backward()        
            optimizer.step()       

            total_loss += loss.item()
            n_batches += 1
            if i % LOG_EVERY_N_BATCHES == 0:
                logger.info("Época %d: batch %d/%d, loss=%.4f", epoch, i, len(train_loader), loss.item())
        logger.info("Época %d/%d terminada: loss promedio=%.4f", epoch, N_EPOCHS, total_loss / max(n_batches, 1))

    logger.info("Prediciendo y calculando score sobre test")
    pred, real, word_ids = _predict(model, test_loader, feature_extractor, device)
    return model.cpu(), problem.metric.score(pred, real, word_ids)
