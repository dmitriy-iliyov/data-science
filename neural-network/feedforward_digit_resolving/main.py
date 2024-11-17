import numpy as np
import tensorflow_datasets as tfds

from model import Model


def load_data():
    (ds_train, ds_test), ds_info = tfds.load('mnist', split=['train', 'test'], as_supervised=True, with_info=True)
    return ds_train, ds_test


def prepared_data():
    ds_train, ds_test = load_data()
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for image, label in tfds.as_numpy(ds_train):
        x_train.append(image)
        y_train.append(label)
    for image, label in tfds.as_numpy(ds_test):
        x_test.append(image)
        y_test.append(label)

    return (np.expand_dims(np.array(x_train), axis=-1) / 255.0, np.array(y_train),
            np.expand_dims(np.array(x_test), axis=-1) / 255.0, np.array(y_test))


train_data, train_labels, test_data, test_labels = prepared_data()

model = Model()
model.fit(train_data, train_labels, epochs=5, batch_size=32)
model.predict(test_data, test_labels, 10)
