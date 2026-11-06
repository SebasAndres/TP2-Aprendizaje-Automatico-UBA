from typing import Any

from sklearn.ensemble import RandomForestClassifier

from src.metrics import PunctuationMetric
from src.problem.base import Problem
from src.problem.punctuation.bidirectional_rnn import create_bidirectional_rnn_model
from src.problem.punctuation.classic import create_classic_model
from src.problem.punctuation.standard_rnn import create_standard_rnn_model


class PunctuationProblem(Problem):
    metric = PunctuationMetric()

    def classic_model(self) -> RandomForestClassifier:
        return create_classic_model()

    def standard_rnn_model(self) -> Any:
        return create_standard_rnn_model()

    def bidirectional_rnn_model(self) -> Any:
        return create_bidirectional_rnn_model()
