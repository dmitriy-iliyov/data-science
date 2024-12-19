import os

import numpy as np
from PIL import Image
from xception.model import CustomizedXception


def get_img(img_path):
    image = Image.open(img_path)
    image = image.convert('RGB')
    resized_image = image.resize((299, 299))
    image_array = np.array(resized_image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


video_path = '/Users/sayner/github_repos/data-science/neural-network/xception/video/test_video.mp4'

xception_model = CustomizedXception()
xception_model.load_model()

for i in os.listdir('/Users/sayner/github_repos/data-science/neural-network/xception/img'):
    r = xception_model.predict(
        get_img(os.path.join('/Users/sayner/github_repos/data-science/neural-network/xception/img', i)))
    print(i + ' - ' + r)

timestamps = xception_model.analyze_video(video_path)
print(timestamps)
