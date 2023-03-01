#!/bin/env python3

"""
Dimensionality reduction with kernel PCA
"""

import os

import numpy as np
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

X = np.load('data.npy')
y = np.load('labels.npy')

storm = y == 1
no_storm = np.invert(storm)

kpca = KernelPCA(n_components=2, kernel='rbf')
X2 = kpca.fit_transform(X)

plt.title('2d Kernel PCA Dimensionality Reduction')
plt.scatter(X2[no_storm, 0], X2[no_storm, 1], color='blue', label='no storm')
plt.scatter(X2[storm, 0], X2[storm, 1], color='red', label='storm')
plt.legend()
plt.savefig(os.path.join('images', 'kernel_pca_2d.jpg'))

kpca = KernelPCA(n_components=3, kernel='rbf')
X3 = kpca.fit_transform(X)

ax = plt.figure().add_subplot(projection='3d')
ax.set_title('3d Kernel PCA Dimensionality Reduction')
ax.scatter(X3[no_storm, 0], X3[no_storm, 1], X3[no_storm, 2], color='blue', label='no storm')
ax.scatter(X3[storm, 0], X3[storm, 1], X3[storm, 2], color='red', label='storm')
ax.view_init(elev=10, azim=-91, roll=0)
plt.legend()
plt.savefig(os.path.join('images', 'kernel_pca_3d.jpg'))
