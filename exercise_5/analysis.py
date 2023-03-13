#!/bin/env python3

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

metal_bands = pd.read_csv('metal_bands_2017.csv', index_col=0)
world_pop = pd.read_csv('world_population_1960_2015.csv', index_col=0)

## Origin of bands
nb_bands_by_nation = metal_bands['origin'].value_counts().sort_values(ascending=False)

fig = plt.figure()
fig.set_size_inches(8, 6)
fig.subplots_adjust(bottom=0.25)
# https://stackoverflow.com/a/52272617/12864941
ax = fig.add_subplot(111)

ax.bar(nb_bands_by_nation.index[:10], nb_bands_by_nation.values[:10])
ax.tick_params(axis='x', rotation=75)
ax.set_xlabel('Nation')
ax.set_ylabel('Number of bands')
plt.title('Top 10 nations by number of bands')
# plt.show()
plt.savefig(os.path.join('images', 'top10_nation_by_number_of_bands_by_nation.jpg'))

plt.figure(figsize=(8, 6))
plt.bar(nb_bands_by_nation.index, nb_bands_by_nation.values)
plt.xticks([])
plt.tick_params(axis='x', bottom=False)
plt.ylabel('Number of bands')
plt.title('Distribution of metal bands by country')
# plt.show()
plt.savefig(os.path.join('images', 'distribution_number_bands_by_nation.jpg'))

plt.clf()
plt.boxplot(nb_bands_by_nation)
plt.xticks([])
plt.ylabel('Number of bands')
plt.title('Box plot of the number of bands by nation')
# plt.show()
plt.savefig(os.path.join('images', 'box_plot_number_bands_by_nation.jpg'))
