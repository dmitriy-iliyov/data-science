import time

from keras import Model
from keras.src.applications.inception_v3 import InceptionV3
from keras.src.layers import GlobalAveragePooling2D, Dense
from keras.src.optimizers import Adam
from matplotlib import pyplot as plt


class CustomInceptionV3:

    def __init__(self, classes_count=5):
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(classes_count, activation='softmax')(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)

        print(self.model.summary())

        self.model.compile(Adam(learning_rate=1e-4),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.store_path = '/content/drive/MyDrive/main/languages/Python/neural_network/labs/lab_5/model'

    def fit(self, train_dataset, val_dataset, epochs=90):
        start = time.time()
        history = self.model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
        execution_time = time.time() - start
        self.model.save(self.store_path + '/model.keras')
        self.plot_history(history, epochs, execution_time)

    def evaluate(self, test_dataset):
        test_loss, test_accuracy = self.model.evaluate(test_dataset)
        print(f"Test accuracy: {test_accuracy:.3f}")

    @staticmethod
    def plot_history(history, epochs, execution_time):
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