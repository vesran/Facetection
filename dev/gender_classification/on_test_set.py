import tensorflow as tf
import os.path as op
import os
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def imshow(img):
    plt.imshow(img, interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()


DATA_DIR = op.join('/', 'datascience', 'datasets', 'faces', 'UTKFace_test')
filenames = os.listdir(DATA_DIR)

# Read all images
print('--- Read images and extract labels')
images = []
genders = []
for filename in tqdm(filenames):
    face = cv2.imread(op.join(DATA_DIR, filename), cv2.IMREAD_COLOR)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (128, 128)) / 255.0
    arr = filename.split('_')
    genders.append(int(arr[1]))
    images.append(face)

images = np.array(images)
genders = np.array(genders)


# Model
print('--- Loading model')
model = tf.keras.models.load_model('./serialized/model_gender_perso_060920.h5')

print('--- Evaluate')
X_test = images
y_test = tf.keras.backend.one_hot(genders, 2)
loss, acc = model.evaluate(X_test, y_test)

print(f'Loss : {loss}\nAcc : {acc}')
