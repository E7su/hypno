# !/usr/lib/python3.4
# -*- coding: utf-8 -*-

from numpy import linspace
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
from answer import create_answer_file

data = load_boston()  # sklearn.datasets.load_boston()

sign = data['data']
classes = data['target']  # target vector

# Bring signs in the sample to the same scale using sklearn.preprocessing.scale function.
sign = scale(sign)

# Используйте KNeighborsRegressor с n_neighbors=5 и weights='distance' — данный параметр добавляет в алгоритм веса,
#  зависящие от расстояния до ближайших соседей.
# В качестве метрики качества используйте среднеквадратичную ошибку
#  (параметр scoring='mean_squared_error' у cross_val_score).
# Качество оценивайте, как и в предыдущем задании, с помощью кросс-валидации по 5 блокам с random_state = 42.

best_p = None
best_score = None

# Different variants of the parameter p metrics on the grid from 1 to 10
# with a step to a total of 200 options were tested (use numpy.linspace function).
for p in linspace(1, 10, 200):

    # This option adds to the weight algorithm, depending on the distance to the nearest neighbors.
    clf = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
    clf.fit(sign, classes)

    # The cross-validation method for the k nearest neighbors
    cv_accuracy = KFold(len(classes), n_folds=5, random_state=42)

    # As a quality metric use the mean squared error
    scores = cross_val_score(clf, sign, classes, cv=cv_accuracy, scoring='mean_squared_error')

    # Function 'cross_val_score' returns an array of signs of the fold's quality.
    #  It is necessary to maximize the average of these folds
    if best_score is None or best_score < scores.mean():
        best_p = p  # value of p to obtain optimum quantity
        best_score = scores.mean()

best_p = round(best_p, 2)

create_answer_file(best_p)
