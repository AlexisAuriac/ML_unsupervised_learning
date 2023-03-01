#!/bin/env python3

"""
This code is largely inspired from this tutorial: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
Computes and plots a silhouette analysis with hierarchical clustering.
"""

import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm

X = np.load('data.npy')
Z = linkage(X, method='weighted')

MIN_K = 2
MAX_K = 8
k_silhouette_list = []

for i, n_clusters in enumerate(range(MIN_K, MAX_K + 1)):
    ## Compute hierarchical clusters and silhouette analysis
    cluster_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
    centers = np.array([X[cluster_labels == cluster].mean(axis=0) for cluster in range(n_clusters)])

    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f'For n_clusters = {n_clusters} the average silhouette score is: {silhouette_avg:.3f}')
    k_silhouette_list.append([n_clusters, silhouette_avg])

    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    ## Plot silhouette analysis
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title('Silhouette plot')
    ax1.set_xlabel('Silhouette coefficient values')
    ax1.set_ylabel('Cluster label')

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color='red', linestyle='--', label=f'silhouette average ({silhouette_avg:.3f})')
    ax1.legend()

    ax1.set_yticks([]) # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1]) # We don't really show negative values

    ## Plot the clustering
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title('Clustered data')
    ax2.set_xlabel('1st feature')
    ax2.set_ylabel('2nd feature')

    plt.suptitle(
        f'Silhouette analysis for Hierarchical clustering with n_clusters = {n_clusters}',
        fontsize=14,
        fontweight='bold',
    )

    plt.savefig(os.path.join('images', f'hierarchical_silhouette(k={n_clusters}).jpg'))


df = pd.DataFrame.from_records(k_silhouette_list, columns=['k', 'silhouette average'])
print(df.sort_values(by='silhouette average', ascending=False).to_string(index=False))
