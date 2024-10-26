from abc import ABC

from abc import abstractmethod


class Cleaner(ABC):

    def __init__(self):
        self._sample = None
        self._cleaned_sample = None

    @abstractmethod
    def clean(self, sample, percent, q=None):
        pass

    @abstractmethod
    def info(self):
        pass
