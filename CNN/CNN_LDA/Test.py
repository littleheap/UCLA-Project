import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_redundant=0, n_classes=5, n_informative=3,
                           n_clusters_per_class=1, class_sep=0.5, random_state=10)
fig = plt.figure()

ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)

plt.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o', c=y)

fig.show()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=10)

lda.fit(X, y)

X_new = lda.transform(X)

print(X_new.shape)

plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)

plt.show()
