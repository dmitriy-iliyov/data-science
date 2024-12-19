import time

import numpy as np
from keras import Model
from keras.src.applications.xception import Xception
from keras.src.callbacks import EarlyStopping
from keras.src.layers import GlobalAveragePooling2D, Dense
from keras.src.optimizers import Adam
from keras.src.saving import load_model
from matplotlib import pyplot as plt
from tensorflow_datasets.core.features.image_feature import cv2


class CustomizedXception:

    def __init__(self):
        self._model = None
        self._image_size = (299, 299)

    def init(self):
        base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(2, activation='softmax')(x)

        self._model = Model(inputs=base_model.input, outputs=predictions)
        for layer in base_model.layers:
            layer.trainable = False

        self._model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        self._model.summary()

    def fit(self, train_data, train_answers, validation_split=0.2, epochs=100, batch_size=32):
        start_time = time.time()
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=1,
            restore_best_weights=True
        )
        history = self._model.fit(
            train_data, train_answers,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping]
        )
        execution_time = time.time() - start_time
        print(f"Training completed in {execution_time:.2f} seconds.")
        self.plot_history(history, execution_time)
        self.save()
        return history, execution_time

    def evaluate(self, test_data, test_answers):
        test_loss, test_accuracy = self._model.evaluate(test_data, test_answers, verbose=1)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
        return test_loss, test_accuracy

    def predict(self, sequence):
        _map = {0: 'logo', 1: 'not a logo'}
        _id = np.argmax(self._model.predict(sequence)[0])
        return _map.get(_id)

    def save(self, path='/content/drive/MyDrive/main/languages/Python/neural_network/labs/lab_7/model/lstm_model.keras'):
        self._model.save(path)

    def load_model(self, path='/Users/sayner/github_repos/data-science/neural-network/cnn-bi-lstm/model_files/cnn_bi_lstm_model.keras'):
        self._model = load_model(path)

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

        frame_count = 0
        timestamps = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % int(frame_rate) == 0:
                resized_frame = cv2.resize(frame, self._image_size)
                img_array = np.expand_dims(resized_frame / 255.0, axis=0)
                prediction = self._model.predict(img_array)
                print(prediction)
                if np.argmax(prediction[0]) == 1:
                    timestamps.append(frame_count / frame_rate)

        cap.release()
        return timestamps

    @staticmethod
    def plot_history(history, execution_time):
        epochs = len(history.history['loss'])
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(range(1, epochs + 1), history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy (Execution Time: {execution_time:.2f} seconds)')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(range(1, epochs + 1), history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()
