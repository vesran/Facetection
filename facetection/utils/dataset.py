import tensorflow as tf
import numpy as np
import skimage.io as io
import os
import yaml
from tqdm import tqdm
from glob import glob


with open('./parameters.yaml', 'r') as f:
    params = yaml.full_load(f)
    assert len(params) > 0


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_utkface_record(input_dir, output_file):
    writer = tf.io.TFRecordWriter(output_file)
    filenames = glob(os.path.join(input_dir, '*'))

    for img_path in tqdm(filenames):
        img = np.array(io.imread(img_path))

        labels = img_path.split(os.sep)[-1].split('_')
        if len(labels) != 4:
            print(f'[INFO] incompatible name : {img_path}')
            continue
        age = int(labels[0])
        gender = int(labels[1])
        race = int(labels[2])

        img_raw = img.tostring()

        # Convert to tf.Example
        example = tf.train.Example(features=tf.train.Features(feature={
            'age': _int64_feature(age),
            'gender': _int64_feature(gender),
            'race': _int64_feature(race),
            'image_raw': _bytes_feature(img_raw),
        }))

        # Write to file
        writer.write(example.SerializeToString())

    writer.close()


def read_utkface_tfrecord(tfrecords_path):
    raw_image_dataset = tf.data.TFRecordDataset(tfrecords_path)

    # Create a dictionary describing the features.
    image_feature_description = {
        'age': tf.io.FixedLenFeature([], tf.int64),
        'gender': tf.io.FixedLenFeature([], tf.int64),
        'race': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

    return parsed_image_dataset


def parse_data(features, label=None):
    """ MapDataset to model's input : image, label
    :param features:
    :param label:
    :return:
    """
    image_raw = features['image_raw'].numpy()
    img_1d = np.frombuffer(image_raw, dtype=np.uint8)
    reconstructed_img = img_1d.reshape((params['face_resolution'], params['face_resolution'], -1))
    return reconstructed_img, features[label]


if __name__ == '__main__':
    write_utkface_record(params['validation_set'], './serialized/validation_set.tfrecords')
    write_utkface_record(params['training_set'], './serialized/training_set.tfrecords')
