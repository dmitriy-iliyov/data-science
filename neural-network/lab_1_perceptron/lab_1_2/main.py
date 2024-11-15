import numpy as np
import perceptron as perceptron_package
from perceptron import Perceptron
import time


def prepare_data_for_4x_xor(split=True):
    variables = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], [0, 1])).T.reshape(-1, 4)

    xor_result = np.bitwise_xor(np.bitwise_xor(variables[:, 0], variables[:, 1]),
                                np.bitwise_xor(variables[:, 2], variables[:, 3]))
    if not split:
        return variables, xor_result

    return np.array(variables[:12]), np.array(xor_result[:12]), np.array(variables[12:]), np.array(xor_result[12:])


def print_prepared_data():
    print("Training data:\n", train_data)
    print("Training answers:\n", train_answers)
    # print("Test data:\n", test_data)
    # print("Training answers:\n", test_answers)


train_data, train_answers = prepare_data_for_4x_xor(False)

perceptron = Perceptron()
# print(perceptron.get_hidden_layer_weights())
# print(perceptron.get_output_layer_weights())
perceptron.train(train_data, train_answers, 1000, 0.001)
p_answers = [perceptron.xor(i) for i in train_data]
for i, answers in enumerate(p_answers):
    print(f"{train_data[i]} : {answers}")
perceptron.save_perceptron()

# perceptron = perceptron_package.load_perceptron()
# print(perceptron.get_hidden_layer_weights())
# print(perceptron.get_output_layer_weights())
