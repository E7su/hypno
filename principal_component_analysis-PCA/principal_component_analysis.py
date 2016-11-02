# !/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
from sklearn.decomposition import PCA
from numpy import corrcoef
from answer import create_answer_file

prices = pandas.read_csv('close_prices.csv')
X = prices.loc[:, 'AXP':]

# In the downloaded data educate PCA transformation with the number of components = 10.
pca = PCA(n_components=10)
pca.fit(X.values)

var = 0
n_var = 0
# The field contains explained_variance_ratio_ percentage of variance that explains each component
for v in pca.explained_variance_ratio_:
    n_var += 1
    var += v
    if var >= 0.9:
        break
# How many component will suffice to explain 90% of the variance?
create_answer_file(1, n_var)

# Apply the constructed transformation to the source data and take the value of the first component.
df_comp = pandas.DataFrame(pca.transform(X))
comp0 = df_comp[0]

# Download the information about the index of Dow Jones of djia_index.csv file.
data = pandas.read_csv('djia_index.csv')
dji = data['^DJI']
corr = corrcoef(comp0, dji)
# What is the Pearson correlation between the first component and the index of Dow Jones?
create_answer_file(2, corr[1, 0])

# Components_ field contains information on the contributions made by the signs in the components.
com = pandas.Series(pca.components_[0])
comp_top = com.sort_values(ascending=False).head(1).index[0]
# Which company has the most weight in the first component?
company = X.columns[comp_top]
create_answer_file(3, company)
