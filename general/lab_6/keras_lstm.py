
import time

from keras.src.callbacks import EarlyStopping
from keras.src.saving import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM
from matplotlib import pyplot as plt


class KerasLSTM:

    def __init__(self):
        self._model = None

    def init(self):
        self._model = Sequential([
            Input(shape=(24, 1)),
            LSTM(128, activation='tanh', return_sequences=True),
            LSTM(128, activation='tanh', return_sequences=False),
            Dense(1, activation='linear')
        ])
        self._model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    def summary(self):
        self._model.summary()

    def fit(self, train_data, train_answers, val_data=None, val_answers=None, epochs=100, batch_size=32):
        start_time = time.time()
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=1,
            restore_best_weights=True
        )
        history = self._model.fit(
            train_data, train_answers,
            validation_data=(val_data, val_answers) if val_data is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[early_stopping]
        )
        execution_time = time.time() - start_time
        print(f"Training completed in {execution_time:.2f} seconds.")
        self.plot_history(history, epochs, execution_time)
        return history, execution_time

    def evaluate(self, test_data, test_answers):
        test_loss, test_mae = self._model.evaluate(test_data, test_answers, verbose=1)
        return test_loss, test_mae

    def load_model(self, path='/Users/sayner/github_repos/data-science/general/lab_6/data_files/models/lstm_bitcoin_model.keras'):
        self._model = load_model(path)

    def predict(self, sequence):
        return self._model.predict(sequence)

    @staticmethod
    def plot_history(history, epochs, execution_time):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), history.history['mae'], label='Training MAE')
        if 'val_mae' in history.history:
            plt.plot(range(1, epochs + 1), history.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error')
        plt.title(f'MAE (Execution Time: {execution_time:.2f} seconds)')
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
