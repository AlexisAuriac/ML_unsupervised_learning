# Part 5: application of unsupervised learning

## Subject

Pick a dataset and perform an unsupervised learning on it. Your dataset has to be different from any dataset seen during the course. Ideally, your algorithm should answer an interesting question about the dataset. The unsupervised learning can then be either a clustering, a dimensionality reduction or a regression.

You are free to choose the dataset within the following constraints :
- several hundreds of lines
- at least 6 attributes (columns), the first being a unique id
- some features may be categorical (non quantitative).

If necessary, you can tweak an existing dataset in order to artificially make it possible to apply analysis ans visualization techniques. Example resources to find datasets:
- [Link 1](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research)
- [Link 2](https://perso.telecom-paristech.fr/eagan/class/igr204/datasets)
- [Link 3](https://github.com/awesomedata/awesome-public-datasets)
- [Link 4](https://www.kaggle.com/datasets)

You could start with a general analysis of the dataset, with for instance a file ```analysis.py``` that studies:
- histograms of quantitative variables with a comment on important statistical aspects, such as means, standard deviations, etc...
- A study of potential outliers
- Correlation matrices (maybe not for all variables)
- Any interesting analysis: if you have categorical data, which categories are represented most ? To what extent ?

If the dataset is very large you may also extract a random sample of the dataset to build histogram or compute correlations. You can discuss whether the randomness of the sample has an important influence on the analysis result (this will depend on the dataset)

Whether it is a clustering, a dimensionality reduction or a density estimation, you should provide an evaluation of your processing. This can for instance be:
- for a clustering, it can be an inertia, a normalized cut...
- for a dimensionality reduction, the explained variance
- for a density estimation, the kullbach leibler divergence between the dataset and a dataset sampled from the estimated distribution
- but you are encouraged to use other evaluations if they are more relevant for your processing.

Short docstrings in the python files will be appreciated, at least at the beginning of each file.

In our report, you could include for instance:
- general information on the dataset found in the analysis file.
- a potential comparison between several algorithm / models that you explored, if relevant
- a presentation of the method used to tune the algorithms (choice of hyperparameters, cross validation, etc).
- a short discussion of the results

Feel free to add useful visualizations for each step of your processing.

## Main sources

[Fun, beginner-friendly datasets](https://www.kaggle.com/code/rtatman/fun-beginner-friendly-datasets)

[Is Metal Music Dying?](https://www.kaggle.com/code/guyabihanna/is-metal-music-dying)

## Solution

We chose a dataset about metal bands by nation from 1960 to 2017 ([source](https://www.kaggle.com/datasets/mrpantherson/metal-by-nation)).

The data was scraped from the website [metalstorm](http://metalstorm.net/)

### Clean up

(see convert_dataset.py)

The dataset's encoding was ISO-8859-1, we changed it to utf-8.

Some bands appeared twice in the dataset.

### Analysis

The dataset has 6 features:
- origin: country of origin of the band
- band_name
- fans: number of fans of the band
- formed: year the band formed
- split: year the band split, if applicable
- style: style(s) of the band

The dataset was collected from the [metalstorm](http://metalstorm.net/) website, its data may be heavily skewed towards certain bands, genres, etc...

In particular when it comes to the year the bands were formed, it may be the case that older bands are underrepresented. Considering that the older bands go back to the 60s, those bands are probably past their prime, their fans use the internet less than that of newer bands, or they might just be dead. On the other hand the older bands that are represented are those that are remembered, the most popular ones.

#### Origin

![Top 10 nations by number of bands](images/top10_nation_by_number_of_bands_by_nation.jpg?raw=true)

![Distribution of metal bands by country](images/distribution_number_bands_by_nation.jpg?raw=true)

![Box plot of the number of bands by nation](images/box_plot_number_bands_by_nation.jpg?raw=true)

*Points are outliers for this box plot*

The distribution of bands is very unequal with the large majority of nations having only 1 bands (43 out of 113), because of this most of the nations are considered outliers.

#### Fans

![Top 10 bands by number of fans](images/top10_bands_by_number_of_fans.jpg?raw=true)

![Number of fans by band](images/fans_hist.jpg?raw=true)

![Number of fans by band (log scale)](images/fans_hist_log.jpg?raw=true)

The distribution of fans is very unequal with the large majority of bands having less than 10 fans (2265 out of 5000), because of this a lot of the more interesting bands are outliers.

#### Formation date

![Formation date histogram](images/formed_hist.jpg?raw=true)

4 bands don't have a formation date.

The average formation date is ```2000.52``` and the its standard deviation is ```8.83```.

#### Split date

![Split date histogram](images/split_hist.jpg?raw=true)

2189 out of 4941 bands don't have a split date.

The average split date is ```2001.34``` and the its standard deviation is ```8.90```.

The average and standard deviation of formation and split are suspiciously similar, this may be symptomatic of the way the data was collected.

![Band active period histogram](images/active_period_hist.jpg?raw=true)

2468 out of 2752 bands split the same year they were formed.

Active period average: ```1.08``` years

Active period standard deviation: ```3.78``` years

The large majority of bands split the year they were formed, this doesn't seem right. After checking a few bands it seems that the split date is often the same as the formed date for no good reason.

### Clustering

For the unsupervised learning part of this exercise we chose to do a clustering based on the number of fans and the year the band was formed.

This part isn't as detailed as we wanted it to be because of the poor quality of the dataset and lack of time.

We used the Kmeans clustering with the knee method.

![Clustering of bands based on formation date and number of fans (kmeans/knee)](images/kmeans_knee_clustering.jpg?raw=true)

The data is clustered in what looks like "levels of popularity" with the year it was formed having little impact on the clusters.
