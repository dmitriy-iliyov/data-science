from img_analyzer import ImgAnalyzer
from PIL import Image


# resized_image.save('/Users/sayner/github_repos/neural-network/lab_5/files/resized_image.jpg')

img_an = ImgAnalyzer()
print(img_an.analyze('/Users/sayner/github_repos/neural-network/lab_5/files/test_img_6.jpg'))
