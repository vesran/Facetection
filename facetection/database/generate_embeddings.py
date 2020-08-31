"""
Creates embeddings for a image using FaceNet.
It contains functions to embed a single face as well as all images in a specified directory.
"""


import cv2
import os
from glob import glob
import pickle
import params
from tqdm import tqdm
import face_recognition


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

    embeddings = face_recognition.face_encodings(face)
    return embeddings[0] if len(embeddings) > 0 else None


def pickle_dataset(outpath='./serialized/faces_embeddings.pkl'):
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
        for image in tqdm(images):
            print(image)
            embedding = embed_face(image)
            embeddings.append(embedding) if embedding is not None else 0
        data[label] = embeddings

    # Saving data
    with open(outpath, 'wb') as f:
        pickle.dump(data, f)
    return data


if __name__ == '__main__':
    # filename = './datasets/Joey/011_JoeyTribbiani_20200810.jpg'
    # embed_face(filename)
    pickle_dataset(outpath='./serialized/faces_embeddings.pkl')

    outpath = './serialized/faces_embeddings.pkl'
    with open(outpath, 'rb') as f:
        data = pickle.load(f)

    X = []
    y = []
    for i, name in enumerate(data):
        for e in data[name]:
            X.append(e.tolist())
            y.append(i)

    import numpy as np
    X = np.array(X)
    y = np.array(y)

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pca = PCA(2)
    twodim = pca.fit_transform(X)
    scat = plt.scatter(twodim[:,0], twodim[:,1], c=y)
    legend1 = plt.legend(*scat.legend_elements(), loc="lower left", title="Classes")
