# !/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from answer import create_answer_file

test = pd.read_csv('perceptron-test.csv', header=None)
train = pd.read_csv('perceptron-train.csv', header=None)

X_train = train.ix[:, 1:2].values  # sign (2, 3 column)
y_train = train[0].values  # target (1 column)

X_test = test.ix[:, 1:2].values
y_test = test[0].values

# Educate perceptron with standart parameters and 'random_state=241'.
clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)
print clf

# Calculate the quality (proportion of correctly classified objects, accuracy) obtained by the classifier
# on the test sample.
predictions = clf.predict(X_test)
# print clf.score(X_test, y_test)  # 0.665
accuracy_before = accuracy_score(y_test, predictions)  # 0.665
print 'Accuracy beforenormalization: ', accuracy_before

# Normalize training and a test sample using StandardScaler class.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = Perceptron(random_state=241)  # find a part of right answers in the test sample
clf.fit(X_train_scaled, y_train)  # educate perceptron on new samples
predictions = clf.predict(X_test_scaled)

accuracy_after = accuracy_score(y_test, predictions)  # 0.845
# print clf.score(X_test_scaled, y_test)  # 0.845
print 'Accuracy after normalization: ', accuracy_after

# Find the difference between quality on the test sample after normalization and quality without it.
quality_diff = accuracy_after - accuracy_before
print 'Difference between quality on the test sample after normalization and quality without it: ', quality_diff

create_answer_file(round(quality_diff, 3))
