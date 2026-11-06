from abc import ABC, abstractmethod
from typing import Any

from src.metrics import Metric


class Problem(ABC):
    """Define la métrica y crea modelos nuevos, sin entrenar, para cada tipo de problema"""

    metric: Metric

    @abstractmethod
    def classic_model(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def standard_rnn_model(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def bidirectional_rnn_model(self) -> Any:
        raise NotImplementedError
