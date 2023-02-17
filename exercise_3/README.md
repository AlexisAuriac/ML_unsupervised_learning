# Part 3: company clustering customers

## Subject

A company has gathered data about its customers and would like to identify similar clients, in order to propose relevant products to new clients, based on their features. This can be represented as a clustering problem. The data are stored in ```exercise_3/data.npy```. They are 4 dimensional.

Pick:
- two clustering methods
- two heuristics to choose a relevant number of clusters, and perform different clusterings of this dataset (overall, you have 2 × 2 = 4 methods). You must use a different metric for each clustering method. You could for instance use the standard euclidean metric for one method, and a different metric for the other method, for instance based on a rescaling of the dimensions of the data (hence, you could transform the data first, and apply a known metric on the transformed data.)

Compare and discuss the difference between the results of the different methods you tried. Discuss whether one mehod (combination of the clustering method and of heuristic) seems to give more interesting or clearer results than the others.

You may use libraries such as scikit-learn in order to implement the methods.

## Main sources

[Best Practices for Visualizing Your Cluster Results](https://towardsdatascience.com/best-practices-for-visualizing-your-cluster-results-20a3baac7426)

https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

## Solution

### Kmeans with Knee

![Kmeans with knee](images/kmeans_knee.jpg?raw=true)
