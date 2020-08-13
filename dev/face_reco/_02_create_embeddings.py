import cv2
import numpy as np
import os, yaml
from glob import glob
import pickle


with open('./parameters.yaml', 'r') as f:
    params = yaml.full_load(f)
    assert len(params) > 0

embedder = cv2.dnn.readNetFromTorch(params['face_embedding_model'])


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


directories = glob(os.path.join(params['datasets'], '*'))
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
with open('./serialized/embeddings.pkl', 'wb') as f:
    pickle.dump(data, f)





