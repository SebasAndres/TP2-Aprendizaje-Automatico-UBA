from typing import Any

from sklearn.ensemble import RandomForestClassifier

from src.metrics import CapitalizationMetric
from src.problem.base import Problem
from src.problem.capitalization.bidirectional_rnn import create_bidirectional_rnn_model
from src.problem.capitalization.classic import create_classic_model
from src.problem.capitalization.standard_rnn import create_standard_rnn_model


class CapitalizationProblem(Problem):
    metric = CapitalizationMetric()

    def classic_model(self) -> RandomForestClassifier:
        return create_classic_model()

    def standard_rnn_model(self) -> Any:
        return create_standard_rnn_model()

    def bidirectional_rnn_model(self) -> Any:
        return create_bidirectional_rnn_model()
