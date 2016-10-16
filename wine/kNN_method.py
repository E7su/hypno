# !/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

data = pandas.read_csv('wine.data', index_col=None, header=None)
classes = data[0]  # class recorded is in the first column (three variants)
sign = data.ix[:, 1:].copy()  # sign is in the columns of the second to last (more about sign in wine.name)

# Evaluation of the quality necessary to carry out the method of cross - validation for 5 fold.
# Create partitions generator that mixes the sample before forming units (shuffle = True).
# For reproducible results, create generator KFold fixed parameter random_state = 42.
# As a quality measure using the proportion of correct answers (accuracy).
kf = KFold(len(sign), n_folds=5, shuffle=True, random_state=42)

# Find the classification accuracy on the cross-validation method for
# the k nearest neighbors (sklearn.neighbors.KNeighborsClassifier), with k of the 1 to 50.
cv_accuracy = [
    cross_val_score(estimator=KNeighborsClassifier(n_neighbors=k), X=sign, y=classes, cv=kf).mean()
    for k in range(1, 51)]

print 'Classification accuracy:'
print cv_accuracy
answer = ['', '', '', '']
# Value of k to obtain optimum quantity.
answer[1] = max(cv_accuracy)
answer[0] = cv_accuracy.index(answer[1]) + 1
answer[1] = (round(answer[1], 2))

# Make a scaling signs with uses function sklearn.preprocessing.scale.
scaled_sign = scale(sign)

# Find the optimum value of k on the cross-validation method for the k nearest neighbors.
scaled_cv_accuracy = [
    cross_val_score(estimator=KNeighborsClassifier(n_neighbors=k), X=scaled_sign, y=classes, cv=kf).mean()
    for k in range(1, 51)]

print 'Scaled classification accuracy'
print scaled_cv_accuracy

# Optimal k after bringing signs to the same scale
answer[3] = max(scaled_cv_accuracy)
answer[2] = scaled_cv_accuracy.index(answer[3]) + 1
answer[3] = round(answer[3], 2)


# Function for create file with answer for coursera
def create_answer_file(number):
    answer_file = 'answer_' + str(number) + '.txt'
    value = answer[number]
    file = open(answer_file, 'w')
    file.write(str(value))
    file.close()


i = 0
for i in xrange(4):
    create_answer_file(i)

print 'Answer: сlassification accuracy, k, scaled classification accurancy, scaled k'
print answer
