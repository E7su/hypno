# !/usr/bin/python
# -*- coding: utf-8 -*-

import pandas
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

# We need only Pclass, Fare, Age and Sex
data.drop(['PassengerId', 'SibSp', 'Name', 'Parch', 'Ticket', 'Cabin', 'Embarked'], inplace=True, axis=1,
          errors='ignore')

data = data.dropna(axis=0)  # drop nan values
data = data.replace({'male': 1, 'female': 0})  # string to number
Y = data["Survived"]  # target variable
data.drop(['Survived'], inplace=True, axis=1, errors='ignore')
print data.tail()

clf = DecisionTreeClassifier(random_state=241)
clf.fit(data, Y)  # decision tree learning

importances = clf.feature_importances_  # find feature importance
print importances  # Sex Fare
