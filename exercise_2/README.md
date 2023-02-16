# Part 2

## Subject

A meteorological station has gathered 800 data samples in dimension 6, thanks to 6 sensors. The operators of the station would like to predict the risk of a tempest the next day, but first, they need to reduce the dimensionality of the data, in order to apply a supervised learning algorithm on the reduced data.

The data is stored in the ```exercise_2``` folder:
- ```data.npy``` contains the raw data
- ```labels.npy``` contains the results for each sample: 1 if there is a tempest, 0 otherwise.

Perform a dimensionality reduction of the data, to a dimension of 2 and 3 and plot these reductions onto scatter plots in dimension 2 and 3 as well, coloring the projected samples according to the label of the original sample.

Which dimension, between 2 and 3, seems to allow to predict the label based on the projected components only ?

You may use libraries such as scikit-learn in order to implement your dimensionality reduction method, that you are free to choose (linear or non linear). One of the methods that we have seen during the class works well, with a well chosen output dimension. You are encouraged to try at least one other dimensionality reduction method, and if the results is not as good as the previous method, to present them shortly in your report as well.

## Main sources

[Article on methods of dimensionality reduction](https://towardsdatascience.com/11-dimensionality-reduction-techniques-you-should-know-in-2021-dcb9500d388b)

## Solution

### PCA

(see pca.py)

This method is linear.

This method works really well with 3 dimensions.

![PCA (2d)](images/pca_2d.jpg?raw=true "PCA (2d)")

![PCA (3d)](images/pca_3d.jpg?raw=true "PCA (3d)")

### Kernel PCA

(see kernel_pca.py)

This method is non-linear.

This method does not work well at all.

![Kernel PCA (2d)](images/kernel_pca_2d.jpg?raw=true "Kernel PCA (2d)")

![Kernel PCA (3d)](images/kernel_pca_3d.jpg?raw=true "Kernel PCA (3d)")

### LDA

Briefly tried LDA but it didn't work since it needs:
```
n_components >= min(n_features, n_classes - 1)
```

Our number of features is 6, but the number of classes is 2.

Therefore we can only have a dimensionality of 1.

### Truncate SVD

(see svd.py)

This method is linear.

It gives very similar results to PCA, with 3d working really well.

![SVD (2d)](images/svd_2d.jpg?raw=true "SVD (2d)")

![SVD (3d)](images/svd_3d.jpg?raw=true "SVD (3d)")
