import time

import tensorflow as tf
from tensorflow import keras


class KerasCNN:

    def __init__(self, input_neurons=2, hidden_layer_count=1, hidden_neurons=10):
        self._hidden_layer_count = hidden_layer_count
        self._hidden_neurons = hidden_neurons
        inputs = keras.Input(shape=(input_neurons,))
        hidden_layer = keras.layers.Dense(hidden_neurons, activation='relu')(inputs)
        for _ in range(1, hidden_layer_count):
            hidden_layer = keras.layers.Dense(hidden_neurons, activation='relu')(hidden_layer)
        outputs = keras.layers.Dense(1, activation='linear')(hidden_layer)
        self._cnn = keras.Model(inputs=inputs, outputs=outputs)
        self._cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                          loss='mean_squared_error',
                          metrics=['mean_squared_error', 'accuracy'])

    def fit(self, train_data, train_labels, epochs=100):
        start_time = time.time()
        history = self._cnn.fit(train_data, train_labels, epochs=epochs)
        execution_time = time.time() - start_time

        accuracy_list = history.history['accuracy']
        mse_list = history.history['mean_squared_error']
        epochs = len(accuracy_list)

        statistics = {
            'network': 'KerasCNN',
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
        return [i[0] for i in self._cnn.predict(test_data)]
