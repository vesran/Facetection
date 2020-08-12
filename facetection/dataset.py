import tensorflow as tf
import yaml, os
import pathlib


with open('./parameters.yaml', 'r') as f:
    params = yaml.full_load(f)
    assert len(params) > 0

# TODO : remove fake images
# TODO : write TODO list in README
data_dir = pathlib.Path(params['training_set'])


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  subset="training",
  seed=123,
  image_size=params['face_resolution'],
  batch_size=32)



