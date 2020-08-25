import tensorflow as tf
import os.path as op
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rand
from datetime import datetime
from tensorflow.keras import layers


def imshow(img):
    plt.imshow(img, interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()


LOG_DIR = op.join('logs', 'facetection_images', datetime.now().strftime("%Y%m%d-%H%M%S"))
DATA_DIR = op.join('/', 'datascience', 'datasets', 'faces', 'UTKFace_train')
filenames = os.listdir(DATA_DIR)
rand.shuffle(filenames)

file_writer = tf.summary.create_file_writer(LOG_DIR)

# Read all images
images = []
genders = []
ages = []
for filename in filenames:
    face = cv2.imread(op.join(DATA_DIR, filename), cv2.IMREAD_COLOR)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (32, 32))
    arr = filename.split('_')
    ages.append(int(arr[0]))
    genders.append(int(arr[1]))
    images.append(face)

images = np.array(images)
genders = np.array(genders)
ages = np.array(ages)


# Write a few images
with file_writer.as_default():
    images_examples = images[:25]
    tf.summary.image('Training images example', images_examples, max_outputs=25, step=0)


# Model
def gen_model():
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))

    x = inputs
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(84, 3, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x1 = layers.Dense(64, activation='relu')(x)
    x2 = layers.Dense(64, activation='relu')(x)
    x1 = layers.Dense(1, activation='sigmoid', name='gender_out')(x1)
    x2 = layers.Dense(1, activation='relu', name='age_out')(x2)

    model = tf.keras.models.Model(inputs=inputs, outputs=[x1, x2])

    model.compile(optimizer='Adam', loss=['binary_crossentropy', 'mae'])
    return model


# Splitting data
n = int(len(filenames) * 0.7)
X_train = images[:n]
y_train = [genders[:n], ages[:n]]
X_valid = images[n:]
y_valid = [genders[n:], ages[n:]]

# Training
model = gen_model()
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=8, monitor='val_loss', restore_best_weights=True),
    tf.keras.callbacks.TensorBoard(LOG_DIR, histogram_freq=1)
]
model.fit(X_train, y_train, epochs=100, batch_size=256, validation_data=(X_valid, y_valid),
          callbacks=callbacks, shuffle=True)



