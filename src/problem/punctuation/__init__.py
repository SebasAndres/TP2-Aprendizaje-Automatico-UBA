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
    # 0.1, el del experimento final (notebooks/experiments/final_birnn_dropout_cabeza.ipynb). El barrido había
    # elegido 1.0 (0.644 vs 0.638 sin decay en test, 0.652 vs 0.647 en control), pero con class_weight_power = 0.25
    # el candidato final y el final usaron 0.1. En capitalización queda en 0: con decay empeora en test
    # (ver notebooks/hyperparameters/regularization/regularization_weight_decay.ipynb)
    bidirectional_rnn_weight_decay = 0.1

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

    def bidirectional_rnn_model(self, dropout: float = 0.0, head_dim: int | None = None) -> Any:
        """Crea la RNN bidireccional de puntuación, sin entrenar.

        Parameters
        ----------
        dropout : float, default=0.0
            Probabilidad de dropout sobre la salida de la RNN.
        head_dim : int or None, default=None
            Capa oculta de la cabeza; ``None`` = cabeza lineal.

        Returns
        -------
        Any
            Ver ``create_bidirectional_rnn_model``.
        """
        return create_bidirectional_rnn_model(dropout, head_dim)
