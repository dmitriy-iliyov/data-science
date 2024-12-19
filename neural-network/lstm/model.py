import time

from keras.src.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Dropout
from matplotlib import pyplot as plt


class KerasLSTM:

    def __init__(self, word_count, max_length):
        self._model = Sequential([
            Input(shape=(max_length,)),
            Embedding(input_dim=word_count + 1, output_dim=128, input_length=max_length),
            LSTM(128, activation='tanh', return_sequences=True),
            #Dropout(0.2),
            LSTM(64, activation='tanh', return_sequences=False),
            # Dropout(0.2),
            Dense(5, activation='softmax')
        ])
        self._model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        self._model.summary()

    def fit(self, train_data, train_answers, validation_split=0.1, epochs=100, batch_size=256):
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
        return self._model.predict(sequence)

    def save(self, path='/content/drive/MyDrive/main/languages/Python/neural_network/labs/lab_7/model/lstm_model.keras'):
        self._model.save(path)

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

