import time

import numpy as np
from keras.src.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

from feedforward_cascade_elman.lab_2.tools import plotter


class Model:

    def __init__(self):
        self.model = keras.Sequential([
            Flatten(input_shape=(28, 28, 1)),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        print(self.model.summary())
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, train_data, train_labels, epochs=5, batch_size=32):
        train_labels = keras.utils.to_categorical(train_labels, 10)
        start = time.time()
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        history = self.model.fit(train_data,
                                 train_labels,
                                 batch_size,
                                 epochs=epochs,
                                 validation_split=0.2,
                                 callbacks=[early_stopping])
        execution_time = time.time() - start
        plotter.print_plot_lab_3({'epochs': epochs,
                                  'accuracy': history.history['accuracy'],
                                  'cce': history.history['loss'],
                                  'batch_size': batch_size,
                                  'execution_time': execution_time})

    def predict(self, test_data, test_labels, n):
        test_labels_cat = keras.utils.to_categorical(test_labels, 10)
        print()
        self.model.evaluate(test_data, test_labels_cat)
        predictions = [np.argmax(p) for p in self.model.predict(test_data[:n])]
        for a, p in zip(test_labels[:n], predictions):
            print(f"digit = {a}; pred = {p}")
