import time
from keras import Sequential, Input
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Conv2D, Flatten, Dense, MaxPooling2D, ZeroPadding2D, Dropout
from keras.src.optimizers import Adam
from keras.src.saving.saving_lib import load_model
from matplotlib import pyplot as plt
from inception_v3.img_analyzer import get_img


class AlexNet:

    def __init__(self):
        self.model = None
        self.store_path = '/content/drive/MyDrive/'

    def initialize(self):
        self.model = Sequential([
            Input((224, 224, 3)),
            Conv2D(96, (11, 11), strides=4, padding='valid', activation='relu'),
            MaxPooling2D(pool_size=(3, 3), strides=2),
            ZeroPadding2D(padding=(2, 2)),
            Conv2D(256, (5, 5), strides=1, padding='valid', activation='relu'),
            MaxPooling2D(pool_size=(3, 3), strides=2),
            Conv2D(384, (3, 3), padding='same', activation='relu'),
            Conv2D(384, (3, 3), padding='same', activation='relu'),
            Conv2D(256, (3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(3, 3), strides=2),
            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(200, activation='softmax')
        ])

        self.model.compile(optimizer=Adam(learning_rate=1e-4),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def load(self, path='/Users/sayner/github_repos/data-science/neural-network/alexnet/learned_model/lab_4_model.keras'):
        self.model = load_model(path)

    def print_model(self):
        print(self.model.summary())

    def fit(self, train_dataset, val_dataset, save=False, epochs=20):
        start = time.time()
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        history = self.model.fit(train_dataset,
                                 validation_data=val_dataset,
                                 validation_split=0.2,
                                 epochs=epochs,
                                 callbacks=[early_stopping])
        execution_time = time.time() - start
        if save:
            self.model.save(self.store_path + '/lab_4_model.keras')
        self.plot_history(history, execution_time)

    def evaluate(self, test_dataset):
        loss, accuracy = self.model.evaluate(test_dataset)
        print(f"Loss: {loss}, Accuracy: {accuracy}")

    def predict(self, img_path):
        img_bytes = get_img(img_path)
        probability_vec = list(self.model.predict(img_bytes)[0])
        return probability_vec

    @staticmethod
    def plot_history(history, execution_time):
        epochs = len(history.history['accuracy'])
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), history.history['accuracy'], label='Training Accuracy')
        plt.plot(range(1, epochs + 1), history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy (Execution Time: {execution_time:.2f} seconds)')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), history.history['loss'], label='Training Loss')
        plt.plot(range(1, epochs + 1), history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()

        plt.show()