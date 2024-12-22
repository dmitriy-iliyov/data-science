from abc import ABC
from abc import abstractmethod


class Clusterer(ABC):

    def __init__(self, sample):
        self._sample = sample
        self._clusters = []

    @abstractmethod
    def clusterize(self):
        pass

    @abstractmethod
    def info(self):
        pass
