import os
import shutil
from itertools import islice

from keras.src.legacy.preprocessing.image import ImageDataGenerator
import keras


def train_data(dir_path):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        dir_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        dir_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    return train_datagen, train_generator, validation_generator


def prepare_data(directory, class_count, new_dir):
    dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))][:class_count]
    os.mkdir(new_dir)
    for d in dirs:
        shutil.copy(os.path.join(directory, d), os.path.join(new_dir, d))


def analyze_dir(directory):
    dir_count = 0
    files_count = 0
    dir_path = None
    for d in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, d)):
            if dir_path is None:
                dir_path = os.path.join(directory, d)
            dir_count += 1
    print(dir_path)
    for _ in os.listdir(os.path.join(dir_path, 'images')):
        files_count += 1
    print("classes count: " + str(dir_count))
    print("files count: " + str(files_count))


def get_code_name_map(line):
    splited_s = line.split('\t')
    splited_s[1] = splited_s[1].replace('\n', '')
    return splited_s[0], splited_s[1]


def get_class_names(names_path):
    class_names = {}
    with open(names_path) as file:
        for line in file:
            class_code, class_name = get_code_name_map(line)
            class_names[class_code] = class_name
    return class_names


def get_class_codes(names_path):
    class_codes = []
    with open(names_path) as file:
        for line in file:
            class_codes.append(str(line).strip())
    return class_codes


# dir_path = "/Users/sayner/main/data/tiny-imagenet-200/train"
# analyze_dir(dir_path)
needed_class_codes = get_class_codes('/lab_4/files/wnids.txt')
all_class_names = get_class_names('/lab_4/files/words.txt')
class_code_name_map = {}
classes_indexes = []
for index, code in enumerate(needed_class_codes):
    class_code_name_map[code] = all_class_names[code]
    classes_indexes.append(index)
print(class_code_name_map)
train_lables = keras.utils.to_categorical(classes_indexes, 200)
