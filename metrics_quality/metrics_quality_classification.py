# !/usr/bin/env pythonclassification.py
# -*- coding: utf-8 -*-

import pandas
import sklearn.metrics as metrics
from answer import *

# True classes of objects — column truth
# and answers some of the classifier — column predicted.
data = pandas.read_csv('classification.csv')

# Fill in the table of classification errors:
# true positive  (TP)
# false positive (FP)
# true negative  (TN)
# false negative (FN)
clf_table = {'tp': (1, 1), 'fp': (0, 1), 'fn': (1, 0), 'tn': (0, 0)}
for name, res in clf_table.iteritems():
    clf_table[name] = len(data[(data['true'] == res[0]) & (data['pred'] == res[1])])

create_answer_file(1, '{tp} {fp} {fn} {tn}'.format(**clf_table))

#  Calculate the basic metrics of the quality classifier:

# Accuracy (доля верно угаданных) — sklearn.metrics.accuracy_score
acc = metrics.accuracy_score(data['true'], data['pred'])

# Precision (точность) — sklearn.metrics.precision_score
pr = metrics.precision_score(data['true'], data['pred'])

# Recall (полнотв) — sklearn.metrics.recall_score
rec = metrics.recall_score(data['true'], data['pred'])

# F-measure — sklearn.metrics.f1_score
f1 = metrics.f1_score(data['true'], data['pred'])

create_answer_file(2, '{:0.2f} {:0.2f} {:0.2f} {:0.2f}'.format(acc, pr, rec, f1))

# There are four trained classifier. In scores.csv file recorded true classes and the value of the degree
# of affiliation positive class for each classifier on a sample:
# For the logistic regression - the probability of a positive grade (score_logreg column)
# For SVM - indent from separating surface (column score_svm),
# For the metric algorithm - weighted sum of grades neighbors (score_knn column)
# For the final tree - the proportion of positive objects in the list (score_tree column).
data_2 = pandas.read_csv('scores.csv')

# Calculate the area under the ROC-curve for each classifier.
# What qualifier has the highest metric value AUC-ROC
#                                            (specify the name of the column with the answers of the classifier)?
scores = {}
for clf in data_2.columns[1:]:
    scores[clf] = metrics.roc_auc_score(data_2['true'], data_2[clf])

create_answer_file(3, pandas.Series(scores).sort_values(ascending=False).head(1).index[0])

# What classifier achieves maximum precision at a density of (Recall) is not less than 70%?
# What is the accuracy in this work? Find the point with sklearn.metrics.precision_recall_curve function. 
# It returns three arrays: precision, recall, thresholds. 
# They recorded the accuracy and completeness under certain thresholds, these in the array 'thresholds'. 
# Find the maximum value among the accuracy of those records for which completeness is not less than 0.7.
pr_scores = {}
for clf in data_2.columns[1:]:
    pr_curve = metrics.precision_recall_curve(data_2['true'], data_2[clf])
    pr_curve_df = pandas.DataFrame({'precision': pr_curve[0], 'recall': pr_curve[1]})
    pr_scores[clf] = pr_curve_df[pr_curve_df['recall'] >= 0.7]['precision'].max()

create_answer_file(4, pandas.Series(pr_scores).sort_values(ascending=False).head(1).index[0])
