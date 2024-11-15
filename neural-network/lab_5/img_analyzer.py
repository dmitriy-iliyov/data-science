import numpy as np
from PIL import Image
from keras.src.saving.saving_lib import load_model


def get_img(img_path):
    image = Image.open(img_path)
    resized_image = image.resize((224, 224))
    image_array = np.array(resized_image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


class ImgAnalyzer:

    def __init__(self, model_path='/Users/sayner/github_repos/neural-network/lab_5/learned_model/model.keras'):
        self.model = load_model(model_path)
        self.__labels = ['french_bulldog', 'german_shepherd', 'golden_retriever', 'poodle', 'yorkshire_terrier']

    def analyze(self, img_path):
        img_bytes = get_img(img_path)
        probability_vec = list(self.model.predict(img_bytes)[0])
        answer = self.__labels[probability_vec.index(max(probability_vec))]
        return answer
