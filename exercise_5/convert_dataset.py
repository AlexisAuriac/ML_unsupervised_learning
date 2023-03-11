#!/bin/env python3

import pandas as pd

# https://stackoverflow.com/a/18172249/12864941
metal_bands = pd.read_csv('metal_bands_2017.csv', encoding='ISO-8859-1', index_col=1)
world_pop = pd.read_csv('world_population_1960_2015.csv', encoding='ISO-8859-1', index_col=1)
# world_pop = pd.read_csv('world_population_1960_2015.csv')

# Remove useless redundant second column
metal_bands.drop(metal_bands.columns[0], axis=1, inplace=True)

# https://stackoverflow.com/a/47329203/12864941
metal_bands.to_csv('metal_bands_2017.csv', encoding='utf-8')
world_pop.to_csv('world_population_1960_2015.csv', encoding='utf-8')
