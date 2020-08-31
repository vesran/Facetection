import numpy as np
import tensorflow as tf
import cv2
import pickle
import params
import face_recognition


age_recognizer = tf.keras.models.load_model('./serialized/age_recognizer.h5')
gender_recognizer = tf.keras.models.load_model('./serialized/gender_recognizer_bo.h5')

age_gender_recognizer = tf.keras.models.load_model('./serialized/age_gender_model.h5')

# Load models and encoder
with open('./serialized/name_recognizer/svm.pkl', 'rb') as f:
    name_recognizer = pickle.load(f)
with open('./serialized/name_recognizer/le.pkl', 'rb') as f:
    le_names = pickle.load(f)
embedder = cv2.dnn.readNetFromTorch(params.face_embedding_model)


def get_age_gender(face):
    reshaped_face = cv2.resize(face, (64, 64))
    reshaped_face = np.expand_dims(reshaped_face, axis=0)
    age_arr, gender_arr = age_gender_recognizer.predict(reshaped_face)
    return age_arr[0][0], np.argmax(gender_arr[0])


def get_age(face):
    reshaped_face = cv2.resize(face, (32, 32))
    reshaped_face = np.expand_dims(reshaped_face, axis=0)
    age_prediction = age_recognizer.predict(reshaped_face)
    return int(age_prediction[0][0])


def get_name(face, threshold=0.5):
    """ Classifies a face image according to the existing names in the datasets folder
        :param face: face image
        :param threshold: if the name prediction's confidence is below the specified threshold,
        the face will be labelled as "Unknown"
        :return: tuple consisting of the predicted name and the confidence
        """

    face_embedding = face_recognition.face_encodings(face)

    # Predict
    if len(face_embedding) > 0:
        probas = name_recognizer.predict_proba([face_embedding[0]])[0]
        j = np.argmax(probas)
        proba = probas[j]
        print(f"Name : {proba}")
        pred_name = le_names.classes_[j] if proba > threshold else "Unknown"
        return pred_name

    return "Undetected"


def get_gender(face):
    img = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    reshaped_face = cv2.resize(img, (64, 64))
    reshaped_face = reshaped_face[:, :, np.newaxis]
    reshaped_face = np.expand_dims(reshaped_face, axis=0)
    gender_prediction = gender_recognizer.predict(reshaped_face)
    print('gender : ', gender_prediction)
    return 'Man' if gender_prediction[0][0] < 0.5 else 'Woman'

#
# if __name__ == '__main__':
#     from facetection.face_detection import find_faces
#     import matplotlib.pyplot as plt
#     filename = './images/test_yves.jpg'
#     image = cv2.imread(filename)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (200, 200))
#     faces = find_faces(image)
#
#     print(faces)
#
#     face = image[faces[0][2]:faces[0][3], faces[0][0]:faces[0][1]]
#     out = get_age_gender(face)
#     print(out)

