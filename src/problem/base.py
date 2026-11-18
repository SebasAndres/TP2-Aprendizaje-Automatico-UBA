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
    bidirectional_rnn_dropout : float
        Dropout de ``bidirectional_rnn`` (sobre la salida de la RNN y dentro de la cabeza).
    bidirectional_rnn_head_dim : int or None
        Capa oculta de la cabeza de ``bidirectional_rnn``; ``None`` = cabeza lineal.
    """

    metric: Metric
    bidirectional_rnn_weight_decay: float = 0.0
    bidirectional_rnn_dropout: float = 0.0
    bidirectional_rnn_head_dim: int | None = None

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
    def bidirectional_rnn_model(self, dropout: float = 0.0, head_dim: int | None = None) -> Any:
        """Crea la RNN bidireccional del problema, sin entrenar.

        Parameters
        ----------
        dropout : float, default=0.0
            Probabilidad de dropout sobre la salida de la RNN.
        head_dim : int or None, default=None
            Capa oculta de la cabeza; ``None`` = cabeza lineal.

        Returns
        -------
        Any
            Un ``nn.Module`` con ``forward(x, lengths)``.
        """
        raise NotImplementedError
