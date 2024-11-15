from abc import ABC, abstractmethod


class Network(ABC):

    @abstractmethod
    def predict(self, sample):
        pass

    @abstractmethod
    def fit(self, data, answers, epochs, learning_rate, batch_size):
        pass
