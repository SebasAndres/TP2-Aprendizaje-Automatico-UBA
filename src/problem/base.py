from abc import ABC, abstractmethod
from typing import Any

from src.metrics import Metric


class Problem(ABC):
    """Define la métrica y crea modelos nuevos, sin entrenar, para cada tipo de problema.

    Attributes
    ----------
    metric : Metric
        Métrica y loss del problema.
    bidirectional_rnn_weight_decay : float
        Weight decay de AdamW para ``bidirectional_rnn``, elegido en
        ``notebooks/hyperparameters/regularization/regularization_weight_decay.ipynb``.
    """

    metric: Metric
    bidirectional_rnn_weight_decay: float = 0.0

    @abstractmethod
    def classic_model(self) -> Any:
        """Crea el modelo clásico del problema, sin entrenar.

        Returns
        -------
        Any
            Un estimador de sklearn con ``fit`` y ``predict``.
        """
        raise NotImplementedError

    @abstractmethod
    def standard_rnn_model(self) -> Any:
        """Crea la RNN estándar (unidireccional) del problema, sin entrenar.

        Returns
        -------
        Any
            Un ``nn.Module`` con ``forward(x, lengths)``.
        """
        raise NotImplementedError

    @abstractmethod
    def bidirectional_rnn_model(self, dropout: float = 0.0) -> Any:
        """Crea la RNN bidireccional del problema, sin entrenar.

        Parameters
        ----------
        dropout : float, default=0.0
            Probabilidad de dropout sobre la salida de la RNN.

        Returns
        -------
        Any
            Un ``nn.Module`` con ``forward(x, lengths)``.
        """
        raise NotImplementedError
