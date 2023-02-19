#!/bin/env python3

import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt

data = np.load('data.npy')

MAX_K = 15
inertias = list()

for k in range(1, MAX_K):
    kmeans = KMeans(n_clusters=k, n_init='auto').fit(data)
    
    # inertia = kmeans.inertia_
    # print(f'{k} clusters: inertia = {inertia:.2E}')
    # inertias.append(inertia)

# kneedle = KneeLocator(
#     range(1, MAX_K), inertias, S=1.0, curve='convex', direction='decreasing'
# )
# print(f'\nknee at {kneedle.knee} clusters')
# print(f'knee y: {kneedle.knee_y:.2E}')

# plt.plot(range(1, MAX_K), inertias)
# plt.xticks(range(1, MAX_K))

# plt.axvline(kneedle.knee, 0, 1, linestyle='--', color='r', label='knee')

# plt.xlabel('number of centroids')
# plt.ylabel('inertia')
# plt.legend()
# plt.show()
