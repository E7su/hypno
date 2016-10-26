# !/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
from sklearn.svm import SVC
from answer import create_answer_file

df = pandas.read_csv('svm-data.csv', header=None)
y = df[0]  # target
X = df.loc[:, 1:]  # signs

# Teach classifier with linear kernel, with parameters: C = 100000 and random_state = 241.
# This value should be used to ensure that the SVM is working with a sample as a linearly separable sample.
# At lower values ​​of the algorithm will be adjusted taking into account the term of the functional
# fine for small indentation, which is why the result can not match up with the solution
# of the classical problem of SVM for linearly separable sample.
model = SVC(kernel='linear', C=100000, random_state=241)
model.fit(X, y)

# Find the number of objects that are supporting (numbered from one).
n_sv = model.support_
n_sv.sort()
create_answer_file(' '.join([str(n + 1) for n in n_sv]))
