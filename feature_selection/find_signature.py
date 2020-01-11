#!/usr/bin/env python3

import pickle
import numpy
from time import time

numpy.random.seed(42)


"""The words (features) and authors (labels), already largely processed.
These files should have been created from the previous (Lesson 10)
mini-project."""

words_file = "../text_learning/your_word_data.pkl"
authors_file = "../text_learning/your_email_authors.pkl"
# words_file = "../tools/word_data.pkl"
# authors_file = "../tools/email_authors.pkl"

with open(authors_file, 'rb') as file:
    authors = pickle.load(file)

with open(words_file, 'rb') as file:
    word_data = pickle.load(file)

"""test_size is the percentage of events assigned to the test set (the
remainder go into training)
feature matrices changed to dense representations for compatibility with
classifier functions in versions 0.15.2 and earlier"""

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(word_data,
                                                                            authors,
                                                                            test_size=0.1,
                                                                            random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


"""a classic way to overfit is to use a small number
of data points and a large number of features;
train on only 150 events to put ourselves in this regime"""
#features_train = features_train.toarray()
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]

# your code goes here

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=40)

tf0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-tf0, 3), "s")
labels_train_pred = clf.predict(features_train)

tp0 = time()
labels_test_pred = clf.predict(features_test)
print("prediction time:", round(time()-tp0, 3), "s")

print(len(features_train[0]))

from sklearn.metrics import accuracy_score

accuracy_train = accuracy_score(labels_train, labels_train_pred)
print(accuracy_train)

accuracy_test = accuracy_score(labels_test, labels_test_pred)
print(accuracy_test)

imp = clf.feature_importances_
print(imp.max(), imp.argmax())
feat = vectorizer.get_feature_names()
print(feat[imp.argmax()])