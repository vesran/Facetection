"""
Creates embeddings for a image using FaceNet.
It contains functions to embed a single face as well as all images in a specified directory.
"""


import cv2
import os
from glob import glob
import pickle
import params

embedder = cv2.dnn.readNetFromTorch(params.face_embedding_model)


def embed_face(face_or_path):
    # Consider image or path -> image
    if type(face_or_path) == str:
        face = cv2.imread(face_or_path)
    else:
        face = face_or_path

    (fH, fW) = face.shape[:2]
    # ensure the face width and height are sufficiently large
    if fW < 20 or fH < 20:
        return

    # construct a blob for the face ROI, then pass the blob
    # through our face embedding model to obtain the 128-d
    # quantification of the face
    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                     (96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)
    vec = embedder.forward()
    return vec.flatten()


def pickle_dataset(outpath='./serialized/embeddings.pkl'):
    """ Create a dictionary with names/dirs names as keys and list of embeddings as values.
    It pickles the mentioned dictionary.
    :return: dictionary
    """
    directories = glob(os.path.join(params.dataset, '*'))
    data = {}
    for directory in directories:
        label = directory.split(os.path.sep)[-1]
        images = glob(os.path.join(directory, '*'))
        print(f"Embeddings for {label} -- {len(images)} images to go.")
        embeddings = []
        for image in images:
            embedding = embed_face(image)
            embeddings.append(embedding)
        data[label] = embeddings

    # Saving data
    with open(outpath, 'wb') as f:
        pickle.dump(data, f)
    return data





