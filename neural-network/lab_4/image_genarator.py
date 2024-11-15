import tensorflow as tf
import os


def custom_image_generator(labels_map, images_dir):
    class_names = sorted(set(labels_map.values()))
    class_to_index = {name: index for index, name in enumerate(class_names)}
    labels_map = {key: class_to_index[value] for key, value in labels_map.items()}

    image_paths = [os.path.join(images_dir, filename) for filename in labels_map.keys()]
    labels = [labels_map[filename] for filename in labels_map.keys()]

    def load_image(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: load_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset
