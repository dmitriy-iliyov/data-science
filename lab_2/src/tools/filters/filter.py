from abc import ABC, abstractmethod


class Filter(ABC):

    def __init__(self):
        self._sample = None
        self._filtered_sample = []
        self._filtered_speed_list = []

    @abstractmethod
    def filter(self, sample, k_max):
        pass

    @abstractmethod
    def info(self):
        pass
