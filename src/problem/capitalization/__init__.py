from typing import Any

from sklearn.ensemble import RandomForestClassifier

from src.metrics import CapitalizationMetric
from src.problem.base import Problem
from src.problem.capitalization.bidirectional_rnn import create_bidirectional_rnn_model
from src.problem.capitalization.classic import create_classic_model
from src.problem.capitalization.standard_rnn import create_standard_rnn_model


class CapitalizationProblem(Problem):
    """Problema de capitalización: su métrica y sus modelos."""

    metric = CapitalizationMetric()

    def classic_model(self) -> RandomForestClassifier:
        """Crea el modelo clásico de capitalización, sin entrenar.

        Returns
        -------
        RandomForestClassifier
            Ver ``create_classic_model``.
        """
        return create_classic_model()

    def standard_rnn_model(self) -> Any:
        """Crea la RNN estándar de capitalización, sin entrenar.

        Returns
        -------
        Any
            Ver ``create_standard_rnn_model``.
        """
        return create_standard_rnn_model()

    def bidirectional_rnn_model(self, dropout: float = 0.0) -> Any:
        """Crea la RNN bidireccional de capitalización, sin entrenar.

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
