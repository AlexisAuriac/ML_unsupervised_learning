#!/bin/env python3

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from kneed import KneeLocator
import matplotlib.pyplot as plt
import matplotlib.cm as cm

X = np.load('data.npy')
Z = linkage(X, method='weighted')

plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode='level', p=3)
plt.xlabel('Sample index')
plt.ylabel('Distance')

plt.title('Dendrogram of the hierarchical clustering')
plt.axhline(y=500, color='r', linestyle='--', label='k=6')
plt.legend()
plt.savefig('images/hierarchical_dendrogram.jpg')