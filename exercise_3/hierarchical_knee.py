#!/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.datasets import make_blobs
from kneed import KneeLocator

import matplotlib.cm as cm
import matplotlib.pyplot as plt

X = np.load('data.npy')
Z = linkage(X, method='weighted')

k_range = range(3, 9)
clusterings = []

for k in k_range:
    labels = fcluster(Z, k, criterion='maxclust')
    centers = np.array([X[labels == cluster].mean(axis=0) for cluster in range(1, k + 1)])
    ssd = np.sum((X - centers[labels - 1])**2)
    clusterings.append({
        'k': k,
        'labels': labels,
        'centers': centers,
        'ssd': ssd,
    })

def plot_knee(ax, ssds, knee): 
    ax.plot(k_range, ssds)
    ax.set_xticks(k_range)

    ax.axvline(knee, 0, 1, linestyle='--', color='r', label='knee')

    ax.set_title('Knee')
    ax.set_xlabel('Number of centroids')
    ax.set_ylabel('Inertia')
    ax.legend()


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

    ax2.set_title('Clustered data (only first 2 features)')
    ax2.set_xlabel('1st feature')
    ax2.set_ylabel('2nd feature')


ssd_values = [clustering['ssd'] for clustering in clusterings]
kneedle = KneeLocator(k_range, ssd_values, curve='convex', direction='decreasing')
# Find the corresponding clustering (source: https://stackoverflow.com/a/48140611/12864941)
clustering = next(filter(lambda clust: clust['k'] == kneedle.knee, clusterings))

print(f'\nknee at {kneedle.knee} clusters')

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
plot_knee(ax1, ssd_values, kneedle.knee)
plot_clusters(ax2, X, clustering['centers'], clustering['labels'])
plt.suptitle('Hierarchical clustering using knee method heuristic')
plt.show()
