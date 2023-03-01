#!/bin/env python3

"""
1. Creates a multivariate normal distribution
2. Plots n sample
3. Compute the empirical average of the first n samples, as a function of the number of samples n and plot the euclidean distance to the expected value as a function of n
"""

import os

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

## Create the random variable
joint_dist = stats.multivariate_normal([175, 75], [[20, 10], [10, 10]])
E_x = joint_dist.mean[0]
E_y = joint_dist.mean[1]
expected_value = np.array([E_x, E_y])

print(f'Expected value of X: {E_x}')
print(f'Expected value of Y: {E_y}')
print(f'Expected value of Z: {expected_value}')

## Take a sample and plot it
sample_size = 1000
sample = joint_dist.rvs(size=sample_size)

plt.scatter(sample[:, 0], sample[:, 1])
plt.title(f'Sample from the law Z (n={sample_size})')
plt.xlabel('X (height in cm)')
plt.ylabel('Y (weight in kg)')
plt.savefig('images/sample.jpg')
plt.clf()

## Compute and plot the euclidian distance between the empirical average and the expected value
n_values = np.arange(1, sample_size + 1)

# https://stackoverflow.com/a/35662005/12864941
empirical_avg = (sample.cumsum(axis=0).T / n_values).T

euclidean_distances = np.sqrt(np.sum((empirical_avg - expected_value) ** 2, axis=1))

plt.plot(n_values, euclidean_distances)
plt.title('Euclidean distance between E and the empirical average')
plt.xlabel('n')
plt.ylabel('Euclidian distance')
plt.savefig(os.path.join('images', 'euclidian_distance.jpg'))
