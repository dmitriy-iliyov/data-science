import time

import tensorflow as tf
import numpy as np
from tensorflow.python.trackable.base import Trackable

import tools.files as filer
from network import Network


class FNN(Network, Trackable):

    def __init__(self, input_neurons=2, hidden_layer_count=1, hidden_neurons=10):
        self._hidden_layer_count = hidden_layer_count
        self._hidden_neurons_count = hidden_neurons
        self._hidden_w_list = []
        self._hidden_b_list = []
        previous_neurons_count = input_neurons
        for i in range(hidden_layer_count):
            self._hidden_w_list.append(
                tf.Variable(tf.random.uniform([previous_neurons_count, hidden_neurons], -1, 1), dtype=tf.float32))
            previous_neurons_count = hidden_neurons
            self._hidden_b_list.append(tf.Variable(tf.zeros([hidden_neurons]), dtype=tf.float32))
        self._output_w = tf.Variable(tf.random.uniform([hidden_neurons, 1], -1, 1), dtype=tf.float32)
        self._output_b = tf.Variable(tf.zeros([1]), dtype=tf.float32)
        self.deviation = None

        self.checkpoint = tf.train.Checkpoint(
            model=self,
            optimizer=tf.optimizers.Adam()
        )
        self.checkpoint_dir = "checkpoints/fnn_model/"
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)

    def predict(self, x_y):
        hidden_layer_output = tf.expand_dims(x_y, 0)
        for i in range(self._hidden_layer_count):
            hidden_layer_output = self._activation_relu(tf.matmul(hidden_layer_output, self._hidden_w_list[i])
                                                        + self._hidden_b_list[i])
        output = tf.matmul(hidden_layer_output, self._output_w) + self._output_b
        return output.numpy()[0]

    def _fit_forward(self, batch, expand=False):
        if expand:
            hidden_layer_output = tf.expand_dims(batch, 0)
        else:
            hidden_layer_output = batch
        for i in range(self._hidden_layer_count):
            hidden_layer_output = self._activation_relu(tf.matmul(hidden_layer_output, self._hidden_w_list[i])
                                                        + self._hidden_b_list[i])
        output = tf.matmul(hidden_layer_output, self._output_w) + self._output_b
        return output

    def _compute_mse(self, output, y):
        return tf.reduce_mean(tf.square(output - y))

    def fit(self, train_data, train_answers, epochs=1000, learning_rate=0.05, batch_size=1):
        print(f"\033[34mFNN:\033[0m\n - hidden layer count: {self._hidden_layer_count}"
              f"\n - hidden neurons count: {self._hidden_neurons_count}\n")
        start_time = time.time()

        self.deviation = max(train_answers)/100
        optimizer = tf.optimizers.Adam(learning_rate)
        mse_list = []
        accuracy_list = []

        for epoch in range(1, epochs + 1):
            epoch_mse = []
            epoch_accuracy_numerator = 0

            indexes = np.random.permutation(len(train_data))
            train_data = train_data[indexes]
            train_answers = train_answers[indexes]

            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i + batch_size]
                batch_answers = train_answers[i:i + batch_size]
                with (tf.GradientTape() as tape):
                    output = self._fit_forward(batch_data)
                    mse = self._compute_mse(output, batch_answers)
                    epoch_mse.append(mse)

                for _ in range(len(batch_answers)):
                    correct_predictions = np.abs(output.numpy()[_] - batch_answers[_]) < self.deviation
                    epoch_accuracy_numerator += np.sum(correct_predictions)

                gradients = tape.gradient(mse, [*self._hidden_w_list, *self._hidden_b_list, self._output_w, self._output_b])
                optimizer.apply_gradients(zip(gradients, [*self._hidden_w_list, *self._hidden_b_list, self._output_w, self._output_b]))

            mean_mse = tf.reduce_mean([mse.numpy() for mse in epoch_mse])
            mse_list.append(mean_mse.numpy())
            epoch_accuracy = epoch_accuracy_numerator / len(train_data)
            accuracy_list.append(epoch_accuracy)
            if epoch % (epochs // 10) == 0:
                print(f"epoch {epoch:3}/{epochs}, "
                      f"mse={mean_mse:.10f}, "
                      f"accuracy={epoch_accuracy}")
        print()
        execution_time = time.time() - start_time
        if sum(accuracy_list[-3:])/3 > 0.9:
            self.save_model()
        statistic = {'network': 'FNN',
                     'accuracy': accuracy_list,
                     'mse': mse_list,
                     'epochs': epochs,
                     'batch_size': batch_size,
                     'execution_time': execution_time,
                     'hidden_layer_count': self._hidden_layer_count,
                     'hidden_neurons_count': self._hidden_neurons_count}
        filer.save_json(
            '/Users/sayner/github_repos/neural-network/feedforward_cascade_elman/lab_2/data_files/statistics/fnn_statistic.txt',
            statistic)
        return statistic

    def _activation_relu(self, x):
        return tf.maximum(0.0, x)

    def save_model(self):
        save_path = self.checkpoint.save(file_prefix=self.checkpoint_dir + 'fnn_model')
        print(f"\033[35msaved to: {save_path}\033[0m\n")

    def load_model(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        print("\033[35mrestored!\033[0m\n")
