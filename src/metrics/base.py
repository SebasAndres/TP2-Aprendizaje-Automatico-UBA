from abc import abstractmethod

class Metric:

    @abstractmethod
    def score(pred, real):
        raise NotImplementedError