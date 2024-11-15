import numpy as np

from xor_model import XorModel


def prepare_data():
    variables = np.array(np.meshgrid([0, 1], [0, 1])).T.reshape(-1, 2)
    xor_result = np.bitwise_xor(variables[:, 0], variables[:, 1])
    return np.array(variables, dtype=np.float32), np.array(xor_result, dtype=np.float32)


def prepare_data_for_4x_xor(split=True):
    variables = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], [0, 1])).T.reshape(-1, 4)
    xor_result = np.bitwise_xor(np.bitwise_xor(variables[:, 0], variables[:, 1]),
                                np.bitwise_xor(variables[:, 2], variables[:, 3]))
    indices = np.random.permutation(len(variables))

    variables = variables[indices]
    xor_result = xor_result[indices]

    if not split:
        return np.array(variables, dtype=np.float32), np.array(xor_result, dtype=np.float32)

    return (np.array(variables[:12], dtype=np.float32), np.array(xor_result[:12], dtype=np.float32),
            np.array(variables[12:], dtype=np.float32), np.array(xor_result[12:], dtype=np.float32))


# train_data, train_answers = prepare_data()

train_data, train_answers = prepare_data_for_4x_xor(False)
model = XorModel(4, 16)
model.fit(train_data, train_answers, 1000)
predictions = [model.forward(i) for i in train_data]
test_data_p = train_data.astype(np.int32)
test_answers_p = train_answers.astype(np.int32)
for i, prediction in enumerate(predictions):
    print(f"{test_data_p[i]} -> {test_answers_p[i]} p:{prediction}")
