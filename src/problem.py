from abc import ABC, abstractmethod
from typing import Any

from sklearn.ensemble import RandomForestClassifier

from src.metrics import (
    Metric,
    CapitalizationMetric,
    PunctuationMetric
)


class Problem(ABC):
    """Define la métrica y crea modelos nuevos, sin entrenar, para cada tipo de problema"""

    metric: Metric

    @abstractmethod
    def classic_model(self) -> Any:
        raise NotImplementedError

    def standard_rnn_model(self) -> Any:
        raise NotImplementedError

    def bidirectional_rnn_model(self) -> Any:
        raise NotImplementedError


class CapitalizationProblem(Problem):
    metric = CapitalizationMetric()

    def classic_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


class PunctuationProblem(Problem):
    metric = PunctuationMetric()

    def classic_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
