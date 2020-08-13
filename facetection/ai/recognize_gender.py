import tensorflow as tf
import numpy as np
import pickle

from facetection.utils.dataset import parse_data, read_utkface_tfrecord


BATCH_SIZE = 32
EPOCHS = 5


#################################################################################
# Prepare data
#################################################################################

training_file = './serialized/training_set.tfrecords'

init_ds = read_utkface_tfrecord(training_file, label='gender')
labelled_ds = init_ds.map(parse_data)
shuffled_ds = labelled_ds.shuffle(128)

batched_ds = shuffled_ds.batch(BATCH_SIZE)

#################################################################################
# Training model
#################################################################################

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', strides=2))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(batched_ds, epochs=EPOCHS)


