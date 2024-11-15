import time

import tensorflow as tf
from tensorflow import keras


class KerasENN:

    def __init__(self, input_neurons=2, hidden_layer_count=1, hidden_neurons=10):
        self._hidden_layer_count = hidden_layer_count
        self._hidden_neurons = hidden_neurons
        self._enn = keras.Sequential()
        self._enn.add(keras.layers.Input(shape=(None, input_neurons)))
        for _ in range(hidden_layer_count):
            self._enn.add(keras.layers.SimpleRNN(hidden_neurons,
                                                 return_sequences=True))
        # self._keras_model.add(
        #     keras.layers.SimpleRNN(hidden_neurons, return_sequences=False))
        self._enn.add(keras.layers.Dense(1, activation='linear'))
        self._enn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                          loss='mean_squared_error',
                          metrics=['mean_squared_error', 'accuracy'])

    def fit(self, train_data, train_labels, epochs=100):
        start_time = time.time()
        train_data = train_data.reshape((train_data.shape[0], 1, train_data.shape[1]))
        history = self._enn.fit(train_data, train_labels, epochs=epochs)
        execution_time = time.time() - start_time

        accuracy_list = history.history['accuracy']
        mse_list = history.history['mean_squared_error']
        epochs = len(accuracy_list)

        statistics = {
            'network': 'KerasENN',
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
        test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))
        return [i[0, 0] for i in self._enn.predict(test_data)]
