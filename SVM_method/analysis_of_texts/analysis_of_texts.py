#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import numpy as np
from sklearn import datasets, grid_search
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
from answer import create_answer_file

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
X = newsgroups.data  # array of texts
y = newsgroups.target  # number of class

# TF_IDF = TF (term frequency) * IDF (inverse document frequency).
# TF - term frequency
# IDF - inverse document frequency

# Calculate the TF-IDF-attributes for all text
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(X)

# Pick the best option is the minimum of the set C [10 ^ -5, 10 ^ -4, ... 10 ^ 4, 10 ^ 5]
# for the SVM with linear kernel (kernel = 'linear') by means of cross-validation for 5 blocks.
grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
model = SVC(kernel='linear', random_state=241)
# The first argument passed to GridSearchCV qualifier, which will be selected for the parameter values,
# the second - Dictionary (dict), specifies the grid parameters for the search.
gs = grid_search.GridSearchCV(model, grid, scoring='accuracy', cv=cv)
gs.fit(vectorizer.transform(X), y)

score = 0
C = 0
for attempt in gs.grid_scores_:
    if attempt.mean_validation_score > score:
        # quality assessment on cross-validation
        score = attempt.mean_validation_score
        # values of the parameters
        C = attempt.parameters['C']

# Educate SVM for the entire sample with an optimal parameter C, found in the previous step.
model = SVC(kernel='linear', random_state=241, C=C)
model.fit(vectorizer.transform(X), y)

# Find the words with the highest absolute value of the weight (the weight is stored in a field near coef_ svm.SVC).
words = vectorizer.get_feature_names()

# The weights of each feature in a trained classifier stored in coef_ field.
coef = pandas.DataFrame(model.coef_.data, model.coef_.indices)
top_words = coef[0].map(lambda w: abs(w)).sort_values(ascending=False).head(10).index.map(lambda i: words[i])
top_words.sort()

create_answer_file(','.join(top_words))
