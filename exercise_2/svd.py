#!/bin/env python3

"""
Dimensionality reduction with truncate SVD
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

X = np.load('data.npy')
y = np.load('labels.npy')

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

storm = y == 1
no_storm = np.invert(storm)

svd = TruncatedSVD(n_components=2, algorithm='randomized')
X2 = svd.fit_transform(X_scaled)

plt.title('2d SVD Dimensionality Reduction')
plt.scatter(X2[no_storm, 0], X2[no_storm, 1], color='blue', label='no storm')
plt.scatter(X2[storm, 0], X2[storm, 1], color='red', label='storm')
plt.legend()
plt.savefig('images/svd_2d.jpg')

svd = TruncatedSVD(n_components=3, algorithm='randomized')
X3 = svd.fit_transform(X_scaled)

ax = plt.figure().add_subplot(projection='3d')
ax.set_title('3d SVD Dimensionality Reduction')
ax.scatter(X3[no_storm, 0], X3[no_storm, 1], X3[no_storm, 2], color='blue', label='no storm')
ax.scatter(X3[storm, 0], X3[storm, 1], X3[storm, 2], color='red', label='storm')
ax.view_init(elev=10, azim=-91, roll=0)
plt.legend()
plt.savefig('images/svd_3d.jpg')
