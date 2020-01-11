#!/usr/bin/env python3

"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
import numpy as np

from feature_format import featureFormat, targetFeatureSplit

dataset = "../final_project/final_project_dataset.pkl"
with open(dataset, 'rb') as file:
    data_dict = pickle.load(file)

# add more features to features_list!
features_list = ["poi", "salary"]

# data = featureFormat(data_dict, features_list)
# to obtain the same numerical results as the course in python 3
data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3,
                                                                            random_state=42)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
labels_pred = clf.predict(features_test)

print("Predicted POI = %d" % labels_pred.sum())
print("Total number of people in the test set = %d" % len(labels_pred))
print("True positives: %d" % sum(np.asarray(labels_test)*labels_pred))

from sklearn.metrics import precision_score, recall_score

prec = precision_score(labels_test,labels_pred)
rec = recall_score(labels_test,labels_pred)

print("The precision score is %f" % prec)
print("The recall score is %f" % rec)