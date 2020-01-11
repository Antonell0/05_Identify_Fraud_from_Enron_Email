#!/usr/bin/env python3

"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

dataset = "../final_project/final_project_dataset.pkl"
with open(dataset, 'rb') as file:
    data_dict = pickle.load(file)

""" first element is our labels, any added elements are predictor
features. Keep this the same for the mini-project, but you'll
have a different feature list when you do the final project."""

features_list = ["poi", "salary"]

# data = featureFormat(data_dict, features_list)
# to obtain the same numerical results as the course in python 3
data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)

"""Creation of the Decision Tree and training"""
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3,
                                                                            random_state=42)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
labels_pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, labels_pred, normalize=True, sample_weight=None)
print(accuracy)
