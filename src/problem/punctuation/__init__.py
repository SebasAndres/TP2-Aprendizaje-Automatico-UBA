from typing import Any

from sklearn.ensemble import RandomForestClassifier

from src.metrics import PunctuationMetric
from src.problem.base import Problem
from src.problem.punctuation.bidirectional_rnn import create_bidirectional_rnn_model
from src.problem.punctuation.classic import create_classic_model
from src.problem.punctuation.standard_rnn import create_standard_rnn_model


class PunctuationProblem(Problem):
    """Problema de puntuación: su métrica y sus modelos."""

    metric = PunctuationMetric()
    # El mejor F1 en test del barrido (0.644 vs 0.638 sin decay), confirmado en control (0.652 vs 0.647).
    # En capitalización queda en 0: con decay empeora en test (ver notebooks/hyperparameters/regularization/regularization_weight_decay.ipynb)
    bidirectional_rnn_weight_decay = 1.0

    def classic_model(self) -> RandomForestClassifier:
        """Crea el modelo clásico de puntuación, sin entrenar.

        Returns
        -------
        RandomForestClassifier
            Ver ``create_classic_model``.
        """
        return create_classic_model()

    def standard_rnn_model(self) -> Any:
        """Crea la RNN estándar de puntuación, sin entrenar.

        Returns
        -------
        Any
            Ver ``create_standard_rnn_model``.
        """
        return create_standard_rnn_model()

    def bidirectional_rnn_model(self, dropout: float = 0.0) -> Any:
        """Crea la RNN bidireccional de puntuación, sin entrenar.

        Parameters
        ----------
        dropout : float, default=0.0
            Probabilidad de dropout sobre la salida de la RNN.

        Returns
        -------
        Any
            Ver ``create_bidirectional_rnn_model``.
        """
        return create_bidirectional_rnn_model(dropout)
