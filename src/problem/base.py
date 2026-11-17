from abc import ABC, abstractmethod
from typing import Any

from src.metrics import Metric


class Problem(ABC):
    """Define la métrica y crea modelos nuevos, sin entrenar, para cada tipo de problema"""

    metric: Metric
    # Weight decay de AdamW para bidirectional_rnn, elegido en notebooks/hyperparameters/regularization/regularization_weight_decay.ipynb
    bidirectional_rnn_weight_decay: float = 0.0

    @abstractmethod
    def classic_model(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def standard_rnn_model(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def bidirectional_rnn_model(self, dropout: float = 0.0) -> Any:
        raise NotImplementedError
