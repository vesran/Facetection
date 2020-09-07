import tensorflow as tf
import os.path as op
import os
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random as rand
from datetime import datetime
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
    inputs = tf.keras.Input(shape=(128, 128, 3))

    ax = inputs

    ax = tf.keras.layers.BatchNormalization(axis=-1)(ax)

    ax = tf.keras.layers.Conv2D(32, 14, activation='relu')(ax)
    ax = tf.keras.layers.Conv2D(64, 7, activation='relu')(ax)
    ax = tf.keras.layers.MaxPooling2D(2)(ax)
    ax = tf.keras.layers.BatchNormalization(axis=-1)(ax)
    ax = tf.keras.layers.Dropout(0.2)(ax)

    ax = tf.keras.layers.Conv2D(64, 5, activation='relu')(ax)
    ax = tf.keras.layers.Conv2D(84, 3, activation='relu')(ax)
    ax = tf.keras.layers.MaxPooling2D(2)(ax)
    ax = tf.keras.layers.BatchNormalization(axis=-1)(ax)
    ax = tf.keras.layers.Dropout(0.2)(ax)

    ax = tf.keras.layers.Conv2D(32, 3, activation='relu')(ax)
    ax = tf.keras.layers.MaxPooling2D(2)(ax)
    ax = tf.keras.layers.BatchNormalization(axis=-1)(ax)
    ax = tf.keras.layers.Dropout(0.2)(ax)

    ax = tf.keras.layers.Flatten()(ax)

    ax = tf.keras.layers.Dense(128, activation='relu')(ax)
    ax = tf.keras.layers.BatchNormalization(axis=-1)(ax)
    ax = tf.keras.layers.Dense(64, activation='relu')(ax)
    ax = tf.keras.layers.Dense(2, activation='softmax', name='gender')(ax)

    outputs = [ax]

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='model')

    lr = 1.e-4
    epochs = 1000
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, decay=lr / epochs),
                  loss=['categorical_crossentropy'],
                  metrics=['acc']
                  )


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
model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_data=(X_valid, y_valid),
          callbacks=callbacks, shuffle=True)


# model.evaluate(X_valid, y_valid)

# Save model
# model.save('./serialized/gender_recognizer.h5')

