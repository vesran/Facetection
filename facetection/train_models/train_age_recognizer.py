import tensorflow as tf
import os.path as op
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rand
from datetime import datetime
from tqdm import tqdm
from tensorflow.keras import layers


def imshow(img):
    plt.imshow(img, interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()


LOG_DIR = op.join('logs', 'facetection_age', datetime.now().strftime("%Y%m%d-%H%M%S"))
DATA_DIR = op.join('/', 'datascience', 'datasets', 'faces', 'UTKFace_train')
filenames = os.listdir(DATA_DIR)
rand.shuffle(filenames)

file_writer = tf.summary.create_file_writer(LOG_DIR)

# Read all images
images = []
ages = []
for filename in tqdm(filenames):
    face = cv2.imread(op.join(DATA_DIR, filename), cv2.IMREAD_COLOR)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (128, 128))
    arr = filename.split('_')
    ages.append(int(arr[0]))
    images.append(face)

images = np.array(images)
ages = np.array(ages)


# Write a few images
with file_writer.as_default():
    images_examples = images[:25]
    tf.summary.image('Training data examples', images_examples, max_outputs=25, step=0)


# Model
def gen_model():
    inputs = tf.keras.Input(shape=(128, 128, 3))

    ax = inputs

    ax = tf.keras.layers.BatchNormalization(axis=-1)(ax)

    ax = tf.keras.layers.Conv2D(32, 12, activation='relu', padding='same')(ax)
    ax = tf.keras.layers.Conv2D(32, 12, activation='relu', padding='same')(ax)
    ax = tf.keras.layers.MaxPooling2D(2)(ax)
    ax = tf.keras.layers.BatchNormalization(axis=-1)(ax)
    ax = tf.keras.layers.Dropout(0.2)(ax)

    ax = tf.keras.layers.Conv2D(32, 5, activation='relu', padding='same')(ax)
    ax = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(ax)
    ax = tf.keras.layers.MaxPooling2D(2)(ax)
    ax = tf.keras.layers.BatchNormalization(axis=-1)(ax)
    ax = tf.keras.layers.Dropout(0.2)(ax)

    ax = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(ax)
    ax = tf.keras.layers.MaxPooling2D(2)(ax)
    ax = tf.keras.layers.BatchNormalization(axis=-1)(ax)
    ax = tf.keras.layers.Dropout(0.2)(ax)

    ax = tf.keras.layers.Flatten()(ax)

    ax = tf.keras.layers.Dense(128, activation='relu')(ax)
    ax = tf.keras.layers.BatchNormalization(axis=-1)(ax)
    ax = tf.keras.layers.Dense(64, activation='relu')(ax)
    ax = tf.keras.layers.BatchNormalization(axis=-1)(ax)
    ax = tf.keras.layers.Dense(32, activation='relu')(ax)
    ax = tf.keras.layers.Dense(1, activation='relu', name='age')(ax)

    outputs = [ax]

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='model')

    lr = 1.e-4
    epochs = 1000
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, decay=lr / epochs),
                  loss=['mae'],
                  )
    return model


# Splitting data
n = int(len(filenames) * 0.7)
X_train = images[:n]
y_train = ages[:n]
X_valid = images[n:]
y_valid = ages[n:]

# Training
model = gen_model()
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    tf.keras.callbacks.TensorBoard(LOG_DIR, histogram_freq=1)
]
model.fit(X_train, y_train, epochs=100, batch_size=256, validation_data=(X_valid, y_valid),
          callbacks=callbacks, shuffle=True)


# Save model
# model.save('./serialized/age_recognizer.h5')
