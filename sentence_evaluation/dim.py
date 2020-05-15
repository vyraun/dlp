import numpy as np
import pickle
from sklearn.decomposition import PCA
import sys

Glove = {}

f = open(sys.argv[2])
dim = int(sys.argv[1])

print("Loading the vectors.")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    Glove[word] = coefs
f.close()

print("Reading Done.")

X_train = []
X_train_names = []

for x in Glove:
    X_train.append(Glove[x])
    X_train_names.append(x)

X_train = np.asarray(X_train)

pca =  PCA(n_components = 300)
X_train = X_train - np.mean(X_train)
X_fit = pca.fit_transform(X_train)
U1 = pca.components_

for u in U1:
    print("Starting Dimension {}".format(d))
    z = []

    # Removing Projections on Top Components
    for i, x in enumerate(X_train):
        x = np.dot(u.transpose(), x) * u
        z.append(x)
    
    embedding_file = open('embedding_{}.txt'.format(d), 'w')

    for i, x in enumerate(X_train_names):
        embedding_file.write("%s " % x)
        for t in z[i]:
            embedding_file.write("%f " % t)
        embedding_file.write("\n")

    embedding_file.close()
    
print("Loading the vectors.")
