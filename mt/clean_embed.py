import numpy as np
from sklearn.decomposition import PCA
import subprocess

Glove = {}
f = open('wiki.be.vec')

print("Loading Glove vectors.")

i = 0
for line in f:
    if i != 0:
        word, vec = line.split(' ', 1)
        Glove[word] = np.fromstring(vec, sep=' ')
    i += 1
f.close()

print("Done.")
X_train = []
X_train_names = []
for x in Glove:
        X_train.append(Glove[x])
        X_train_names.append(x)

X_train = np.asarray(X_train)
pca_embeddings = {}

# PCA to get Top Components
pca =  PCA(n_components = 300)
X_train = X_train - np.mean(X_train)
X_fit = pca.fit_transform(X_train)
U1 = pca.components_

z = []

# Removing Projections on Top Components
for i, x in enumerate(X_train):
    for u in U1[0:4]:  # 4 to be removed as per the gudelines
        x = x - np.dot(u.transpose(),x) * u
        z.append(x)

z = np.asarray(z)
X_train = z - np.mean(z)
X_new_final = X_train

final_pca_embeddings = {}
embedding_file = open('clean_pca_embed.txt', 'w')

for i, x in enumerate(X_train_names):
        final_pca_embeddings[x] = X_new_final[i]
        embedding_file.write("%s " % x)

        for t in final_pca_embeddings[x]:
                embedding_file.write("%f " % t)

        embedding_file.write("\n")
