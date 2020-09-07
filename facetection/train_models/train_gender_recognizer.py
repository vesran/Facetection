import tensorflow as tf
import os.path as op
import os
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random as rand
from datetime import datetime
from tensorflow.keras import layers
import pickle


def imshow(img):
    plt.imshow(img, interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()


LOG_DIR = op.join('logs', 'facetection_gender', datetime.now().strftime("%Y%m%d-%H%M%S"))
DATA_DIR = op.join('/', 'datascience', 'datasets', 'faces', 'UTKFace_train')
filenames = os.listdir(DATA_DIR)
rand.shuffle(filenames)

file_writer = tf.summary.create_file_writer(LOG_DIR)

# Read all images
print('--- Read images and extract labels')
images = []
genders = []
for filename in tqdm(filenames):
    face = cv2.imread(op.join(DATA_DIR, filename), cv2.IMREAD_COLOR)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (128, 128))
    face = face[:, :, np.newaxis]
    arr = filename.split('_')
    genders.append(int(arr[1]))
    images.append(face)

images = np.array(images)
genders = np.array(genders)

with open('./serialized/tmp/gender_images_128.pkl', 'wb') as f:
    t = (images, genders)
    pickle.dump(t, f)


# Write a few images
print('--- Write images in Tensorboard')
with file_writer.as_default():
    images_examples = images[:5]
    tf.summary.image('Training images example', images_examples, max_outputs=5, step=0)


# Model
def gen_model():
    inputs = tf.keras.layers.Input(shape=(128, 128, 1))

    x = inputs

    x = layers.Conv2D(32, 4, activation='relu')(x)
    x = layers.Conv2D(64, 4, activation='relu')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Dropout(0.2)(x)

    # x = layers.Conv2D(64, 3, activation='relu')(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.MaxPool2D(2)(x)
    # x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1, activation='sigmoid', name='gender_out')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=[x])

    model.compile(optimizer='Adam', loss=['binary_crossentropy'], metrics=['acc'])
    return model


# Splitting data
print('--- Splitting data')
n = int(len(filenames) * 0.7)
X_train = images[:n]
y_train = genders[:n]
X_valid = images[n:]
y_valid = genders[n:]

# Training
print('--- Create & train model')
model = gen_model()
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=30, monitor='val_acc', restore_best_weights=True),
    tf.keras.callbacks.TensorBoard(LOG_DIR, histogram_freq=1)
]
model.fit(X_train, y_train, epochs=1000, batch_size=64, validation_data=(X_valid, y_valid),
          callbacks=callbacks, shuffle=True)


# model.evaluate(X_valid, y_valid)

# Save model
# model.save('./serialized/gender_recognizer.h5')

