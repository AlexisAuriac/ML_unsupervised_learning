# Part 3: company clustering customers

## Subject

A company has gathered data about its customers and would like to identify similar clients, in order to propose relevant products to new clients, based on their features. This can be represented as a clustering problem. The data are stored in ```exercise_3/data.npy```. They are 4 dimensional.

Pick:
- two clustering methods
- two heuristics to choose a relevant number of clusters, and perform different clusterings of this dataset (overall, you have 2 × 2 = 4 methods).

You must use a different metric for each clustering method. You could for instance use the standard euclidean metric for one method, and a different metric for the other method, for instance based on a rescaling of the dimensions of the data (hence, you could transform the data first, and apply a known metric on the transformed data.)

Compare and discuss the difference between the results of the different methods you tried. Discuss whether one mehod (combination of the clustering method and of heuristic) seems to give more interesting or clearer results than the others.

You may use libraries such as scikit-learn in order to implement the methods.

## Main sources

[Best Practices for Visualizing Your Cluster Results](https://towardsdatascience.com/best-practices-for-visualizing-your-cluster-results-20a3baac7426)

[How to make a silhouette a kmeans silouhette analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)

## Solution

### Kmeans

#### Knee/Elbow method

(see kmeans_knee.py)

We use Kmeans (with sklearn) to make clusters and the knee/elbow method (with Kneedle) to determin the best number of clusters.

![Kmeans with knee](images/kmeans_knee.jpg?raw=true)

The exact knee varies with each run since kmeans initialization is random but using the Kneedle's default sensitivity we usely get a knee at k=6, which is consistent with what we can see when we plot the data.

It also sometimes gives k=5, in this case it merges clusters 3 and 5. On rare occasions it gives k=3 which is very odd.

#### Silhouette analysis

(see kmeans_silhouhette.py)

We test for k in \[2, 8\].

![Kmeans with silhouette (k=2)](images/kmeans_silhouette(k=2).jpg?raw=true)
![Kmeans with silhouette (k=3)](images/kmeans_silhouette(k=3).jpg?raw=true)
![Kmeans with silhouette (k=4)](images/kmeans_silhouette(k=4).jpg?raw=true)
![Kmeans with silhouette (k=5)](images/kmeans_silhouette(k=5).jpg?raw=true)
![Kmeans with silhouette (k=6)](images/kmeans_silhouette(k=6).jpg?raw=true)
![Kmeans with silhouette (k=7)](images/kmeans_silhouette(k=7).jpg?raw=true)
![Kmeans with silhouette (k=8)](images/kmeans_silhouette(k=8).jpg?raw=true)

Here is the silhouette average for each, sorted from highest to lowest.

k | silhouette average
--- | ---
6 | 0.887
5 | 0.793
7 | 0.768
4 | 0.660
8 | 0.645
2 | 0.583
3 | 0.556

k=6 clearly has the highest silhouette average which is a good indicator that it is the best clustering.

In other clustering for 7 and 8 we can see negative values, which likely means that some points have been assigned to the wrong cluster, as they more similar to data points in other clusters.

We can see that clustering for 7 and 8 also have very wide fluctuations in the size of the silhouette plots. These fluctuations are less pronounced but still very noticeable for 2, 3, 4, and 5.

Although 6 clusters seems to be the best, depending on the situation 2, 3, 4, or 5 might be more adapted.

Compared to the knee method this heuristic requires more effort to code and a better understanding of the method and context to interpret it, but it also allows for more nuance.

### Hierarchical clustering

[Hierarchical clustering explained](https://towardsdatascience.com/hierarchical-clustering-explained-e59b13846da8)

too many points to make good dendrogram

#### Knee/Elbow method

(see hierarchical_knee.py)

Hierarchical clustering doesn't have inertia compared to Kmeans, so we are going to use the squared sum of distances (ssd) between each point and its assigned cluster centroid instead.

![Hierarchical clustering with knee](images/hierarchical_knee.jpg?raw=true)

We get a knee at k=5.

#### Silhouette analysis

(see hierarchical_silhouette.py)

Again, we test for k in \[2, 8\].

![Hierarchical clustering with silhouette (k=2)](images/hierarchical_silhouette(k=2).jpg?raw=true)
![Hierarchical clustering with silhouette (k=3)](images/hierarchical_silhouette(k=3).jpg?raw=true)
![Hierarchical clustering with silhouette (k=4)](images/hierarchical_silhouette(k=4).jpg?raw=true)
![Hierarchical clustering with silhouette (k=5)](images/hierarchical_silhouette(k=5).jpg?raw=true)
![Hierarchical clustering with silhouette (k=6)](images/hierarchical_silhouette(k=6).jpg?raw=true)
![Hierarchical clustering with silhouette (k=7)](images/hierarchical_silhouette(k=7).jpg?raw=true)
![Hierarchical clustering with silhouette (k=8)](images/hierarchical_silhouette(k=8).jpg?raw=true)

Here is the silhouette average for each, sorted from highest to lowest.

k | silhouette average
--- | ---
6 | 0.887
7 | 0.800
5 | 0.793
8 | 0.706
4 | 0.684
3 | 0.583
2 | 0.556

The results are overall very similar to what we found with Kmeans.

Again, k=6 yields the best silhouette average which is a good indicator that it is the best clustering, k=7 and k=8 have negative values, and we can see wide fluctuations in all clusterings but k=6.

Compared to the knee method this heuristic requires more effort to code and a better understanding of the method and context to interpret it, but it also allows for more nuance.

#### Dendrogram

(see hierarchical_dendrogram.py)

If we plot a dendrogram we can see that after k=6 the distance between clusters becomes significantly smaller.

![Hierarchical clustering dendrogram](images/hierarchical_dendrogram.jpg?raw=true)

### Other methods we briefly tried

### DBSCAN

Looking at the 5th row of the following examples, it seemed like it would be a good fit for our data.

![comparison of clustering methods](https://scikit-learn.org/stable/_images/sphx_glr_plot_cluster_comparison_001.png)

([source](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html))

[DBSCAN tutorial](https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html)

From what we understand, the algorithm chooses the number of clusters for us, without the use of an heuristic.

The optimization comes from tuning its ```eps``` and ```min_samples``` hyperparameters, which can be done using methods such as grid search or random search.
