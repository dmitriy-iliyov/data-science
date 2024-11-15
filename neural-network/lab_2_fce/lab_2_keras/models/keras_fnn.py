import time

import tensorflow as tf
from tensorflow import keras


class KerasFNN:

    def __init__(self, input_neurons=2, hidden_layer_count=1, hidden_neurons=10):
        self._hidden_layer_count = hidden_layer_count
        self._hidden_neurons = hidden_neurons
        self._fnn = keras.models.Sequential()
        self._fnn.add(keras.Input(shape=(input_neurons,)))
        for i in range(hidden_layer_count):
            self._fnn.add(keras.layers.Dense(hidden_neurons, activation='relu'))
        self._fnn.add(keras.layers.Dense(1, activation='linear'))
        self._fnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss='mean_squared_error',
                          metrics=['mean_squared_error', 'accuracy'])

    def fit(self, train_data, train_labels, epochs=100):
        start_time = time.time()
        history = self._fnn.fit(train_data, train_labels, epochs=epochs)
        execution_time = time.time() - start_time

        accuracy_list = history.history['accuracy']
        mse_list = history.history['mean_squared_error']
        epochs = len(accuracy_list)

        statistics = {
            'network': 'KerasFNN',
            'accuracy': accuracy_list,
            'mse': mse_list,
            'epochs': epochs,
            'batch_size': 1,
            'execution_time': execution_time,
            'hidden_layer_count': self._hidden_layer_count,
            'hidden_neurons_count': self._hidden_neurons
        }
        return statistics

    def predict(self, test_data):
        return [i[0] for i in self._fnn.predict(test_data)]
