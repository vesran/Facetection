import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

with open('./serialized/embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

# Data preprocessing
labels = []
train = []
for label in data:
    for emb in data[label]:
        labels.append(label)
        train.append(emb)

X = np.array(train)
le = LabelEncoder()
y = le.fit_transform(labels)


# Plotting embeddings
pca = PCA(2)
twodim = pca.fit_transform(train)
plt.scatter(twodim[:, 0], twodim[:, 1], c=y, alpha=.8)
plt.show()


# Shuffle data
p = np.random.permutation(X.shape[0])
X = X[p]
y = y[p]


# Training model
model = SVC(probability=True)
model.fit(X, y)


# Save model
with open('./serialized/name_recognizer/svm.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save LabelEncoder
with open('./serialized/name_recognizer/le.pkl', 'wb') as f:
    pickle.dump(le, f)
