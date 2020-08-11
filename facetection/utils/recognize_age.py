import cv2
import pickle
import yaml
import os
import numpy as np


with open('./parameters.yaml', 'r') as f:
    params = yaml.full_load(f)
    assert len(params) > 0


# Load models and encoder
with open('./serialized/name_recognizer/svm.pkl', 'rb') as f:
    name_recognizer = pickle.load(f)
with open('./serialized/name_recognizer/le.pkl', 'rb') as f:
    le_names = pickle.load(f)
embedder = cv2.dnn.readNetFromTorch(params['face_embedding_model'])


def recognize_face_name(face):
    """ Classifies a face image according to the existing names in the dataset folder
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
    pred_name = le_names.classes_[j]
    return pred_name, proba

