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

## Fans
nb_fans = metal_bands['fans'].sort_values(ascending=False)

fig = plt.figure()
fig.set_size_inches(8, 6)
fig.subplots_adjust(bottom=0.25)
# https://stackoverflow.com/a/52272617/12864941
ax = fig.add_subplot(111)

ax.bar(nb_fans.index[:10], nb_fans.values[:10])
ax.tick_params(axis='x', rotation=75)
ax.set_xlabel('Band')
ax.set_ylabel('Number of fans')
plt.title('Top 10 bands by number of fans')
# plt.show()
plt.savefig(os.path.join('images', 'top10_bands_by_number_of_fans.jpg'))

plt.clf()
plt.hist(metal_bands['fans'], bins=40)
plt.xlabel('Number of fans')
plt.ylabel('Number of bands')
plt.title('Number of fans by band')
# plt.show()
plt.savefig(os.path.join('images', 'fans_hist.jpg'))

plt.clf()
plt.hist(metal_bands['fans'], bins=40)
plt.gca().set_yscale('log')
plt.xlabel('Number of fans (log scale)')
plt.ylabel('Number of bands')
plt.title('Number of fans by band (log scale)')
# plt.show()
plt.savefig(os.path.join('images', 'fans_hist_log.jpg'))

print(f'{len(nb_fans[nb_fans < 10])} out of {len(nb_fans)} have less than 10 fans')
