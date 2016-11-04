# !/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas

from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from answer import create_answer_file

# Load the data from the file abalone.csv.
# This dataset in which you want to predict the age of the shells (number of rings) on physical measurements.
data = pandas.read_csv('abalone.csv')

# Convert the sign Sec to numeric: F value should go to -1, I at 0, M 1.
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

# Divide the contents of files in the features and the target variable.
# The last column is written the target variable, in other - signs.
labels = data['Rings']
features = data.ix[:, :-1]

# Teach random forest (sklearn.ensemble.RandomForestRegressor) with a different number of trees:
# 1 to 50 (do not forget to put "random_state = 1" in the constructor).
# For each of the options account the quality of the resulting forest on cross-validation for 5 blocks.
# Use the options "random_state = 1" and "shuffle = True"
# to create a cross-validation generator sklearn.cross_validation.KFold.
# As a quality measure, use the share of correct answers (sklearn.metrics.r2_score).
best_n_est = None
best_score = None
for n_est in range(1, 50):
    clf = RandomForestRegressor(n_estimators=n_est, random_state=1)
    clf.fit(features, labels)

    kf = KFold(len(labels), n_folds=5, random_state=1, shuffle=True)
    scores = cross_val_score(clf, features, labels, cv=kf, scoring='r2')

    # Determine at what minimum amount of random forest trees shows the quality in the cross-validation above 0.52.
    if (best_n_est is None or n_est < best_n_est) and scores.mean() > 0.52:
        best_n_est = n_est
        best_score = scores.mean()

    print str(n_est) + ":  " + str(scores.mean())

print(best_n_est)
create_answer_file(1, best_n_est)
