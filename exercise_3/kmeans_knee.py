#!/bin/env python3

import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data = np.load('data.npy')

MIN_K = 2
MAX_K = 15

def plot_clusters(ax, X, centers, labels):
    colors = cm.nipy_spectral(labels.astype(float) / len(centers))
    ax.scatter(
        X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k'
    )

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        marker='o',
        c='white',
        alpha=1,
        s=200,
        edgecolor='k',
    )
    for i, c in enumerate(centers):
        ax.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

    ax.set_title('Clustered data (only first 2 features)')
    ax.set_xlabel('1st feature')
    ax.set_ylabel('2nd feature')


def plot_knee(ax, inertias, knee): 
    ax.plot(range(MIN_K, MAX_K), inertias)
    ax.set_xticks(range(MIN_K, MAX_K))

    ax.axvline(knee, 0, 1, linestyle='--', color='r', label='knee')

    ax.set_title('Knee')
    ax.set_xlabel('Number of centroids')
    ax.set_ylabel('Inertia')
    ax.legend()


kmeans_list = [KMeans(n_clusters=k, n_init='auto').fit(data) for k in range(MIN_K, MAX_K)]
inertias = [kmeans.inertia_ for kmeans in kmeans_list]

for kmeans in kmeans_list:
    print(f'{kmeans.n_clusters} clusters: inertia = {kmeans.inertia_:.2E}')

kneedle = KneeLocator(range(MIN_K, MAX_K), inertias, curve='convex', direction='decreasing')
kmeans = kmeans_list[kneedle.knee - MIN_K]

print(f'\nknee at {kneedle.knee} clusters')

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
plot_knee(ax1, inertias, kneedle.knee)
plot_clusters(ax2, data, kmeans.cluster_centers_, kmeans.labels_)
plt.suptitle('Kmeans clustering using knee method heuristic')
plt.savefig('images/kmeans_knee.jpg')
