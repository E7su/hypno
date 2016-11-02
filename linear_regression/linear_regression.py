# !/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack
from answer import create_answer_file

train = pandas.read_csv('salary-train.csv')


# Preprocessing:
def text_transform(text):
    text = text.map(lambda t: t.lower())

    # Replace all (without letters and numbers) to the spaces
    # - it will facilitate the further division of the text into words.
    # For such a replacement in text fits the following call: re.sub ( '[^ a-zA-Z0-9]', '', text).
    # Use the method replace in DataFrame, to immediately convert all texts:
    text = text.replace('[^a-zA-Z0-9]', ' ', regex=True)
    return text


# Apply TfidfVectorizer for converting text to feature vectors.
# Leave only those words that are found in at least 5 objects (min_df option in TfidfVectorizer).
vec = TfidfVectorizer(min_df=5)
X_train_text = vec.fit_transform(text_transform(train['FullDescription']))

# Replace the missing columns LocationNormalized and ContractTime to special string 'nan'.
train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)

# Apply DictVectorizer for one-hot-encoding features LocationNormalized and ContractTime.
enc = DictVectorizer()
X_train_cat = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))

# Combine all received signs in a matrix "objects-signs."
# Please note that the template for the text and categorical attributes are sparse.
# To join their columns must use the function scipy.sparse.hstack.X_train = hstack([X_train_text, X_train_cat])
X_train = hstack([X_train_text, X_train_cat])

# Educate ridge regression with alpha = 1 parameter.
# Target variable is written in the column SalaryNormalized.
y_train = train['SalaryNormalized']
model = Ridge(alpha=1)
model.fit(X_train, y_train)

# Build forecasts for two examples of file salary-test-mini.csv.
test = pandas.read_csv('salary-test-mini.csv')
X_test_text = vec.transform(text_transform(test['FullDescription']))
X_test_cat = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([X_test_text, X_test_cat])

y_test = model.predict(X_test)
create_answer_file(1, '{:0.2f} {:0.2f}'.format(y_test[0], y_test[1]))
