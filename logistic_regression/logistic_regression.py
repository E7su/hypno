#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import math
from sklearn.metrics import roc_auc_score
from answer import create_answer_file

# This is a two-dimensional sample target variable which takes the value 1 or -1.
data = pandas.read_csv('data-logistic.csv', header=None)
y = data[0]
X = data.loc[:, 1:]


def fun_w1(w1, w2, y, X, k, C):
    l = len(y)
    S = 0
    for i in xrange(0, l):
        S += y[i] * X[1][i] * (1.0 - 1.0 / (1.0 + math.exp(-y[i] * (w1 * X[1][i] + w2 * X[2][i]))))

    return w1 + (k * (1.0 / l) * S) - k * C * w1


def fun_w2(w1, w2, y, X, k, C):
    l = len(y)
    S = 0
    for i in xrange(0, l):
        S += y[i] * X[2][i] * (1.0 - 1.0 / (1.0 + math.exp(-y[i] * (w1 * X[1][i] + w2 * X[2][i]))))

    return w2 + (k * (1.0 / l) * S) - k * C * w2


# Implement gradient descent for a normal and L2-regularized (with regularization factor of 10) logistic regression.
# Use the step length k = 0.1. In use the vector (0, 0) as an initial approximation.
def grad(y, X, C=0.0, w1=0.0, w2=0.0, k=0.1, err=1e-5):
    i = 0
    i_max = 10000
    w1_new, w2_new = w1, w2

    while True:
        i += 1
        w1_new, w2_new = fun_w1(w1, w2, y, X, k, C), fun_w2(w1, w2, y, X, k, C)
        e = math.sqrt((w1_new - w1) ** 2 + (w2_new - w2) ** 2)

        if i >= i_max or e <= err:
            break
        else:
            w1, w2 = w1_new, w2_new

    return [w1_new, w2_new]


# Start the gradient descent and bring to convergence
# (the Euclidean distance between the vectors of weights on neighboring iterations must be less than 1e-5).
# Limit the number of iterations of the ten thousand.
w1, w2 = grad(y, X)
rw1, rw2 = grad(y, X, 10.0)


# As the quality metrics will use the AUC-ROC (area under the ROC-curve).
# It is designed for binary classification algorithms, outstanding assessment of the object
# belonging to one of the classes.
# In fact, the value of this metric is the aggregation of all the quality parameters of algorithms
# that can be obtained by selecting a threshold for evaluating accessories.

# What is takes AUC-ROC on training without regularization and when using it?
# Please note that roc_auc_score function input must be submitted probability estimates calculated learning algorithms.
# For this, use a sigmoid function: a (x) = 1 / (1 + exp (-w1 x1 - w2 x2)).
def a(X, w1, w2):
    return 1.0 / (1.0 + math.exp(-w1 * X[1] - w2 * X[2]))


y_score = X.apply(lambda x: a(x, w1, w2), axis=1)
y_regscore = X.apply(lambda x: a(x, rw1, rw2), axis=1)

auc = roc_auc_score(y, y_score)
regauc = roc_auc_score(y, y_regscore)

create_answer_file("{:0.3f} {:0.3f}".format(auc, regauc))
