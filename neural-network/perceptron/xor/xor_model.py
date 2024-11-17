import tensorflow as tf
import numpy as np


def relu(x):
    return tf.maximum(0.0, x)


def tanh(x, derivative=False):
    _tanh = np.tanh(x)
    if derivative:
        return 1 - _tanh ** 2
    return _tanh


def sigmoid(x):
    return 1 / (1 + tf.exp(-x))


class XorModel:
    def __init__(self, input_neurons=2, hidden_neurons=16):
        self.hidden_w = tf.Variable(tf.random.uniform([input_neurons, hidden_neurons], -1.0, 1.0), dtype=tf.float32)
        self.hidden_b = tf.Variable(tf.zeros([hidden_neurons]), dtype=tf.float32)
        self.output_w = tf.Variable(tf.random.uniform([hidden_neurons, 1], -1.0, 1.0), dtype=tf.float32)
        self.output_b = tf.Variable(tf.zeros([1]), dtype=tf.float32)

        self.checkpoint = tf.train.Checkpoint(w1=self.hidden_w, b1=self.hidden_b, w2=self.output_w, b2=self.output_b)

    def fit_forward(self, x):
        hidden_layer = relu(tf.matmul(tf.expand_dims(x, 0), self.hidden_w) + self.hidden_b)
        output = sigmoid(tf.matmul(hidden_layer, self.output_w) + self.output_b)
        return output

    def forward(self, x):
        hidden_layer = relu(tf.matmul(tf.expand_dims(x, 0), self.hidden_w) + self.hidden_b)
        output = sigmoid(tf.matmul(hidden_layer, self.output_w) + self.output_b)
        return round(output.numpy()[0, 0])

    def compute_loss(self, output, y):
        return tf.reduce_mean(tf.square(output - y))

    def fit(self, train_data, train_answers, epochs=1000, learning_rate=0.05):
        optimizer = tf.optimizers.SGD(learning_rate)
        loss = None
        batch_size = 1
        for epoch in range(epochs):
            loss_array = []
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i + batch_size]
                batch_answers = train_answers[i:i + batch_size]
                with (tf.GradientTape() as tape):
                    output = self.fit_forward(batch_data)
                    _loss = self.compute_loss(output, batch_answers)
                    loss_array.append(_loss.numpy())
                gradients = tape.gradient(_loss, [self.hidden_w, self.hidden_b, self.output_w, self.output_b])
                optimizer.apply_gradients(zip(gradients, [self.hidden_w, self.hidden_b, self.output_w, self.output_b]))
            if epoch % 100 == 0:
                loss = sum(loss_array) / len(loss_array)
                print(f"Epoch {epoch}, Loss: {loss}")
        if loss < 0.01:
            self.save_model()

    def save_model(self, checkpoint_dir="checkpoints/"):
        save_path = self.checkpoint.save(file_prefix=checkpoint_dir + 'xor_model')
        print(f"saved to: {save_path}")

    def load_model(self, checkpoint_dir="checkpoints/xor_model"):
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("restored!")
