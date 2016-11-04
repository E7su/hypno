# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import pandas
import matplotlib.pyplot as plt

from numpy import argmin
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss

from answer import create_answer_file

# Load the sample of gbm-data.csv file with the pandas and convert it into an array of numpy
# (dataframe's option - value).
# In the first column of the file with the data recorded,
# On the same data educate RandomForestClassifier with the number of trees equal to the number of iterations, which achieved the best quality at the gradient boosting of the preceding paragraph, with random_state = 241 and the rest of the default settings. Which log-loss value is obtained on the test in the random forest? (Do not forget that you need to get the predictions using predict_proba function)a chemical reaction has been or has not been.
# All other column # (d1 - d1776) contain various characteristics of the molecule, such as size, shape, etc.
data = pandas.read_csv('gbm-data.csv').values

labels = data[:, 0]
features = data[:, 1:]

# Divide the sample into training and test using a function with parameters train_test_split:
# test_size = 0.8 and random_state = 241.
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.8, random_state=241)


def get_loss(clf, X, y):
    loss = []
    # Use staged_decision_function method for predicting the quality of
    # the learning and the test sample on each iteration.
    for y_pred in clf.staged_decision_function(X):
        # Calculate and plot the values ​​of log-loss (which can be calculated
        # using sklearn.metrics.log_loss function) on the training and test sample
        # Find the minimum value of the metric and the iteration number at which it is achieved.
        loss.append(log_loss(y, 1.0 / (1.0 + numpy.exp(-y_pred))))

    min_iter = argmin(loss)
    min_loss = loss[min_iter]
    return loss, min_iter, min_loss


# Teach GradientBoostingClassifier with parameters
# n_estimators = 250, verbose = True, random_state = 241
# for lr in [1, 0.5, 0.3, 0.2, 0.1]:  # todo
for lr in [0.2]:
    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=lr)
    clf.fit(X_train, y_train)

    train_loss, train_min_iter, train_min_loss = get_loss(clf, X_train, y_train)
    test_loss, test_min_iter, test_min_loss = get_loss(clf, X_test, y_test)

    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()

    # Find the minimum value of the log-loss and the iteration number at which it is achieved, when learning_rate = 0.2.
    if lr == 0.2:
        res_min_loss = test_min_loss
        res_min_iter = test_min_iter

# How would you describe the quality of graphics on a test sample from a certain iteration:
# Retraining (overfitting) or nedoobuchenie (underfitting)?
res = "overfitting"
print res
create_answer_file(1, res)

res = str(round(res_min_loss, 2)) + " " + str(res_min_iter)
print res
create_answer_file(2, res)

# On the same data educate RandomForestClassifier with the number of trees equal to the number of iterations,
# which achieved the best quality at the gradient boosting of the preceding paragraph,
# with random_state = 241 and the rest of the default settings.
# Which log-loss value is obtained on the test in the random forest?
# (Do not forget that you need to get the predictions using predict_proba function)
clf = RandomForestClassifier(n_estimators=res_min_iter, random_state=241)
clf.fit(X_train, y_train)
y_proba = clf.predict_proba(X_test)[:, 1]
rf_loss = log_loss(y_test, y_proba)

res = str(round(rf_loss, 2))
print res
create_answer_file(3, res)
