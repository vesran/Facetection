import numpy as np
import tensorflow as tf
import cv2


age_recognizer = tf.keras.models.load_model('./serialized/age_recognizer.h5')
gender_recognizer = tf.keras.models.load_model('./serialized/gender_recognizer.h5')


def get_age(face):
    reshaped_face = cv2.resize(face, (32, 32))
    reshaped_face = np.expand_dims(reshaped_face, axis=0)
    age_prediction = age_recognizer.predict(reshaped_face)
    return int(age_prediction[0][0])


def get_name(face):
    return "Patrick"


def get_gender(face):
    reshaped_face = cv2.resize(face, (64, 64))
    reshaped_face = np.expand_dims(reshaped_face, axis=0)
    gender_prediction = gender_recognizer.predict(reshaped_face)
    print('gender : ', gender_prediction)
    return 'Man' if gender_prediction[0][0] < 0.5 else 'Woman'
