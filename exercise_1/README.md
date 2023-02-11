# Part 1: data distribution and the law of large numbers

## Subject

The goal of this exercise is to manipulate a data distribution and to get familiar with the law of large numbers in an informal way.

1. Propose a 2-dimensional random variable Z = (X, Y), with X and Y being two real (∈ R), discrete or continuous random variables. These two variables should represent a quantities of your choice (e.g. the age of the individuals in a population, the color of the eyes of these individuals, ...). Compute the expected value of Z, that must be finite.
2. Sample a number n (of your choice) of points from the law of Z and plot them in a 2 dimensional figure.
3. Compute the empirical average of the first n samples, as a function of the number of samples n and verify that it converges to the expected value, by plotting the euclidean distance to the expected value as a function of n.

**Remark**: you may use simple laws. You could for instance start with a very simple joint distribution, make everything work, and then explore more complex distributions.

## Solution

### 1. Random variable

Our random variable Z is a multivariate normal distribution based on the height and weight of individuals.

The mean of the ditribution is ```(175, 75)``` (175cm, 75kg).

The covariance matrix is:
```python
[[20, 10],
 [10, 10]]
```

Its expected value is ```(175, 75)```.

### 2. Sample

We choose n=1000.

![Sample from the law Z (n=1000)](/images/sample.jpg?raw=true)

### 3. Euclidian distance

![Euclidean distance between the expected and the empirical average](/images/euclidian_distance.jpg?raw=true)

We can see that the empirical average converges to the expected value.
