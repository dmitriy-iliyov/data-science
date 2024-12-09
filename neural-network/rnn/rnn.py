import time

import tensorflow as tf

from feedforward_cascade_elman.lab_2.tools import plotter
from feedforward_cascade_elman.lab_2.tools import filer


class RNN(tf.Module):

    def __init__(self, input_neuron_count, hidden_neuron_count, output_neuron_count, hidden_layer_count):
        super().__init__()

        if hidden_layer_count != len(hidden_neuron_count):
            raise ValueError("hidden_layer_count != len(hidden_neuron_count)")

        self._input_neuron_count = input_neuron_count
        self._hidden_layer_count = hidden_layer_count
        self._hidden_neurons_counts = hidden_neuron_count

        self._hidden_w_list = []
        self._hidden_b_list = []
        self._context_hidden_w_list = []

        previous_neuron_count = self._input_neuron_count

        for i in range(len(self._hidden_neurons_counts)):
            self._hidden_w_list.append(
                tf.Variable(tf.random.uniform([previous_neuron_count, self._hidden_neurons_counts[i]], -1, 1), dtype=tf.float32))
            self._hidden_b_list.append(
                tf.Variable(tf.zeros([self._hidden_neurons_counts[i]]), dtype=tf.float32))
            self._context_hidden_w_list.append(
                tf.Variable(tf.random.uniform([previous_neuron_count, self._hidden_neurons_counts[i]], -1, 1), dtype=tf.float32))
            previous_neuron_count = self._hidden_neurons_counts[i]

        self._output_neuron_count = output_neuron_count
        self._output_w = tf.Variable(tf.random.uniform([self._hidden_neurons_counts[-1], self._output_neuron_count], -1, 1),
                                     dtype=tf.float32)
        self._output_b = tf.Variable(tf.zeros([self._output_neuron_count]), dtype=tf.float32)

        self.deviation = None

        self.optimizer = tf.optimizers.Adam()
        self.checkpoint = tf.train.Checkpoint(model=self, optimizer=self.optimizer)
        self.checkpoint_dir = "checkpoints/rnn_model/"
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)

    def summary(self):
        for i in range(len(self._hidden_w_list)):
            print(f"Recurrent hidden layer №{i+1}: {self._hidden_w_list[i].shape}")
            print(f"    Previous hidden layer №{i+1}: {self._context_hidden_w_list[i].shape}")
        print(f"Output layer: {self._output_w.shape}")

    def fit(self, train_data, train_answers, epochs=100, learning_rate=0.05, batch_size=16):

        start_time = time.time()

        self.deviation = max(train_answers) / 100
        self.optimizer.learning_rate = learning_rate

        mse_list = []
        accuracy_list = []

        for epoch in range(epochs):
            epoch_mse = []
            epoch_accuracy_numerator = 0

            for i in range(0, len(train_data) - batch_size):
                batch_data = train_data[i:i + batch_size]
                batch_answers = train_answers[i:i + batch_size]

                with tf.GradientTape() as tape:
                    output = self._fit_passage(batch_data)
                    mse = self._compute_mse(output, batch_answers)

                trainable_vars = (
                        self._hidden_w_list +
                        self._hidden_b_list +
                        self._context_hidden_w_list
                        +
                        [self._output_w, self._output_b]
                )
                gradients = tape.gradient(mse, trainable_vars)
                self.optimizer.apply_gradients(zip(gradients, trainable_vars))
                epoch_mse.append(mse.numpy())

            mean_mse = tf.reduce_mean(epoch_mse).numpy()
            mse_list.append(mean_mse)
            epoch_accuracy = epoch_accuracy_numerator / len(train_data)
            accuracy_list.append(epoch_accuracy)

            if (epoch + 1) % ((epochs + 1) // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}: MSE={mean_mse:.10f}, Accuracy={epoch_accuracy:.4f}")

        print(f"Fit ended in {time.time() - start_time:.2f} secs.")
        if sum(accuracy_list[-3:]) / 3 > 0.9:
            self.save_model()

        statistic = {
            'network': 'RNN',
            'accuracy': accuracy_list,
            'mse': mse_list,
            'epochs': epochs,
            'batch_size': batch_size,
            'execution_time': time.time() - start_time,
            'hidden_layer_count': self._hidden_layer_count,
            'hidden_neurons_count': str(self._hidden_neurons_counts)
        }
        filer.save_json(
            '/Users/sayner/github_repos/neural-network/lab_6/data_files/statistics/rnn_statistic.txt', statistic)
        self.fit_info(statistic)
        return statistic

    def _fit_passage(self, batch):
        outputs = []
        for _input in batch:
            output = tf.transpose(_input)
            for i in range(self._hidden_layer_count):
                output = self._activation_tanh(
                    tf.matmul(output, self._hidden_w_list[i]) +
                    tf.matmul(output, self._context_hidden_w_list[i]) +
                    self._hidden_b_list[i])
            output = tf.matmul(output, self._output_w) + self._output_b
            outputs.append(output)
        return tf.stack(outputs)

    def _default_passage(self, batch):
        output = batch
        for i in range(self._hidden_layer_count):
            output = tf.Variable((self._activation_tanh(
                tf.matmul(output, self._hidden_w_list[i]) +
                tf.matmul(output, self._context_hidden_w_list[i]) +
                self._hidden_b_list[i])))
        return tf.matmul(output, self._output_w) + self._output_b

    @staticmethod
    def _activation_tanh(x):
        return tf.tanh(x)

    @staticmethod
    def _compute_mse(output, y):
        return tf.reduce_mean(tf.square(output - y))

    def save_model(self):
        save_path = self.checkpoint.save(file_prefix=self.checkpoint_dir + 'rnn_model')
        print(f"\033[35mModel saved : {save_path}\033[0m\n")

    def load_model(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        print("\033[35mModel loaded!\033[0m\n")

    def evaluate(self, test_data, test_answers):
        errors = []
        for i in range(len(test_data)):
            output = self._default_passage(test_data[i])
            error = abs(output - test_answers[i])
            if error < self.deviation:
                errors.append(True)
            else:
                errors.append(False)
        return errors.count(True) / len(errors) * 100

    def predict(self, sequence):
        return self._default_passage(sequence)

    @staticmethod
    def fit_info(statistic):
        plotter.one_fit_statistic(statistic)
