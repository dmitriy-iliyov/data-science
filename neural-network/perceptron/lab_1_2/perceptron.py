import os
import numpy as np
import pickle


class Perceptron:

    def __init__(self, input_layer_neuron_count=4, hidden_layer_neuron_count=16, output_layer_neuron_count=1):
        self.__input_layer_neuron_count = input_layer_neuron_count
        self.__hidden_layer_neuron_count = hidden_layer_neuron_count
        self.__output_layer_neuron_count = output_layer_neuron_count
        self.__set_hidden_layer_weights()
        self.__set_output_layer_weights()
        self.__perceptron_data_file_name = 'network_files/perceptron.pkl'
        self.__train_answers = []

    def __activation_sigmoid(self, x, derivative=False):
        sigmoid = 1 / (1 + np.exp(-x))
        if derivative:
            return sigmoid * (1 - sigmoid)
        return sigmoid

    def __activation_tanh(self, x, derivative=False):
        tanh = np.tanh(x)
        if derivative:
            return 1 - tanh ** 2
        return tanh

    def xor(self, input_data):
        input_layer_sum = np.dot(input_data, self.__hidden_layer_weights)
        hidden_layer_out = [self.__activation_tanh(i) for i in input_layer_sum]
        output_layer_sum = np.dot(hidden_layer_out, self.__output_layer_weights)
        y = self.__activation_sigmoid(output_layer_sum)
        return round(y)

    def xor_for_training(self, input_data):
        input_layer_sum = np.dot(input_data, self.__hidden_layer_weights)
        hidden_layer_out = [self.__activation_tanh(i) for i in input_layer_sum]
        output_layer_sum = np.dot(hidden_layer_out, self.__output_layer_weights)
        y = self.__activation_sigmoid(output_layer_sum)
        return y, hidden_layer_out

    def train(self, train_data, train_answers, epochs=10000, learning_rate=0.001):
        self.__train_answers = train_answers
        for epoch in range(epochs + 1):
            true_answers = 0
            for i in range(len(train_data)):
                y, hidden_layer_out = self.xor_for_training(train_data[i])
                if y != train_answers[i]:
                    error = y - train_answers[i]
                    output_layer_delta = error * self.__activation_sigmoid(y, True)
                    for j in range(self.__hidden_layer_neuron_count):
                        self.__output_layer_weights[j] = (self.__output_layer_weights[j] - learning_rate *
                                                          output_layer_delta * hidden_layer_out[j])
                    for k in range(self.__hidden_layer_neuron_count):
                        hidden_layer_delta = (self.__output_layer_weights * output_layer_delta *
                                              self.__activation_tanh(hidden_layer_out[k], True))
                        for j in range(self.__input_layer_neuron_count):
                            self.__hidden_layer_weights[j] = (self.__hidden_layer_weights[j] - learning_rate *
                                                              hidden_layer_delta)
                            # self.__hidden_layer_weights[j] = (self.__hidden_layer_weights[j] - learning_rate *
                            #                                   hidden_layer_delta * train_data[j])
                else:
                    true_answers += 1
            if epoch % 100 == 0:
                loss = true_answers + 1 / len(self.__train_answers)
                print("epoch:", epoch, "/", epochs, " ", 'loss: ', loss)

    def __set_hidden_layer_weights(self, save=False):
        self.__hidden_layer_weights = np.random.uniform(-1, 1, (self.__input_layer_neuron_count,
                                                                self.__hidden_layer_neuron_count))

        if save:
            if not os.path.exists(self.__perceptron_data_file_name):
                self.__hidden_layer_weights = np.random.uniform(-1, 1, (self.__input_layer_neuron_count,
                                                                        self.__hidden_layer_neuron_count))
                with open(self.__perceptron_data_file_name, 'wb') as file:
                    pickle.dump(self.__hidden_layer_weights, file)
            else:
                with open(self.__perceptron_data_file_name, 'rb') as file:
                    self.__hidden_layer_weights = pickle.load(file)

    def __set_output_layer_weights(self):
        self.__output_layer_weights = np.random.uniform(0, 1, self.__hidden_layer_neuron_count)

    def get_hidden_layer_weights(self):
        return self.__hidden_layer_weights

    def get_output_layer_weights(self):
        return self.__output_layer_weights

    def save_perceptron(self, file_path=None):
        if file_path is None:
            with open(self.__perceptron_data_file_name, 'wb') as file:
                pickle.dump(self, file)
        else:
            with open(file_path, 'wb') as file:
                pickle.dump(self, file)

    def load_perceptron(self, file_path=None):
        if file_path is None:
            with open(self.__perceptron_data_file_name, 'rb') as file:
                return pickle.load(file)
        else:
            with(open(file_path, 'rb')) as file:
                return pickle.load(file)
