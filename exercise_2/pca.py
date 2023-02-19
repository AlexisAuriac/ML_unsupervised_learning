#!/bin/env python3

"""
Dimensionality reduction with PCA
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = np.load('data.npy')
y = np.load('labels.npy')

storm = y == 1
no_storm = np.invert(storm)

X2 = PCA(n_components=2).fit_transform(X)

plt.title('2d PCA Dimensionality Reduction')
plt.scatter(X2[no_storm, 0], X2[no_storm, 1], color='blue', label='no storm')
plt.scatter(X2[storm, 0], X2[storm, 1], color='red', label='storm')
plt.legend()
plt.savefig('images/pca_2d.jpg')

X3 = PCA(n_components=3).fit_transform(X)

ax = plt.figure().add_subplot(projection='3d')
ax.set_title('3d PCA Dimensionality Reduction')
ax.scatter(X3[no_storm, 0], X3[no_storm, 1], X3[no_storm, 2], color='blue', label='no storm')
ax.scatter(X3[storm, 0], X3[storm, 1], X3[storm, 2], color='red', label='storm')
ax.view_init(elev=10, azim=-91, roll=0)
plt.legend()
plt.savefig('images/pca_3d.jpg')
