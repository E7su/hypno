import pandas
import re


data = pandas.read_csv('titanic.csv', index_col='PassengerId')
# column PassengerID create numeration for strings in new DataFrame
print data.head
print '===END OF DATA==='

# ---//Sex//---
male, female = data['Sex'].value_counts()
print 'Male:', male
print 'Female:', female

# ---//Survive//---
rip, sur = data['Survived'].value_counts()
# rip -- if value == 0
# sur -- if value == 1
sur = float(sur)
sur = round(sur / len(data) * 100, 2)
print 'Survive percent: ', sur

# ---//FirstClass//---
three, one, two = data['Pclass'].value_counts()
one = float(one)
one = round((one / len(data) * 100), 2)
print 'First class percent: ', one

# ---//Age//---
# age = data.sort_values(by='Age', ascending=True)    // sort
# print age
age = data.Age.median()
print 'Age median: ', age
age = data.Age.mean()  # average
print 'Age average: ', round(age, 2)

# ---//Correlation//---
pearson = data['SibSp'].corr(data['Parch'], method='pearson')
print 'Correlation (Sibsp, Parch) Pearson: ', round(pearson, 2)

# ---//FirstName//---
fn = data[data['Sex'] == 'female']['Name']


def first_name(name):
    # first word in brackets
    gm = re.search(".*\\((.*)\\).*", name)
    if gm is not None:
        return gm.group(1).split(" ")[0]
    # first word after Mrs. or Miss.
    wn = re.search(".*\\. ([A-Za-z]*)", name)
    return wn.group(1)

# the most common name
name = fn.map(lambda full_name: first_name(full_name)).value_counts().idxmax()
print 'The most common female name: ', name
