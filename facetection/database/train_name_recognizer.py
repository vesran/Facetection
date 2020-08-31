import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


# Import data
outpath = './serialized/faces_embeddings.pkl'
with open(outpath, 'rb') as f:
    data = pickle.load(f)

# Data preprocessing
X = []
y = []
for name in data:
    for e in data[name]:
        X.append(e.tolist())
        y.append(name)

le = LabelEncoder()
X = np.array(X)
y = le.fit_transform(y)


# Model training
model = SVC(probability=True)
model.fit(X, y)

with open('./serialized/name_recognizer/svm.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('./serialized/name_recognizer/le.pkl', 'wb') as f:
    pickle.dump(le, f)

