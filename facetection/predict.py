import numpy as np
import tensorflow as tf
import cv2
import pickle
import params


age_recognizer = tf.keras.models.load_model('./serialized/age_recognizer.h5')
gender_recognizer = tf.keras.models.load_model('./serialized/gender_recognizer.h5')

# Load models and encoder
with open('./serialized/name_recognizer/svm.pkl', 'rb') as f:
    name_recognizer = pickle.load(f)
with open('./serialized/name_recognizer/le.pkl', 'rb') as f:
    le_names = pickle.load(f)
embedder = cv2.dnn.readNetFromTorch(params.face_embedding_model)


def get_age(face):
    reshaped_face = cv2.resize(face, (32, 32))
    reshaped_face = np.expand_dims(reshaped_face, axis=0)
    age_prediction = age_recognizer.predict(reshaped_face)
    return int(age_prediction[0][0])


def get_name(face, threshold=0.6):
    """ Classifies a face image according to the existing names in the datasets folder
        :param face: face image
        :return: tuple consisting of the predicted name and the confidence
        """
    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                     (96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)
    vec = embedder.forward()
    face_embedding = vec.flatten()

    # Predict
    probas = name_recognizer.predict_proba([face_embedding])[0]
    j = np.argmax(probas)
    proba = probas[j]
    print(f"---- {proba}")
    pred_name = le_names.classes_[j] if proba > threshold else "Unknown"
    return pred_name


def get_gender(face):
    reshaped_face = cv2.resize(face, (64, 64))
    reshaped_face = np.expand_dims(reshaped_face, axis=0)
    gender_prediction = gender_recognizer.predict(reshaped_face)
    print('gender : ', gender_prediction)
    return 'Man' if gender_prediction[0][0] < 0.5 else 'Woman'
