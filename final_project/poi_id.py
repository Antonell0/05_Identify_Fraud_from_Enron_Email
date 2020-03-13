#!/usr/bin/env python3

import sys
import pickle
import numpy as np
from sklearn.linear_model import SGDClassifier
from time import time

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi']

POI_label = ['poi']

"""Used features
features_list is a list of strings, each of which is a feature name.
The first feature must be "poi"."""
features_list = POI_label + financial_features + ['to_messages', 'from_poi_to_this_person', 'from_messages',
                                                  'from_this_person_to_poi',
                                                  'shared_receipt_with_poi']  # ['poi', 'salary', 'total_payments', 'loan_advances', 'bonus']  # You will need to use more features

# Load the dictionary containing the dataset
dataset = "final_project_dataset.pkl"
with open(dataset, 'rb') as file:
    data_dict = pickle.load(file)

# Initial exploration
print("The number of people contained in the dataset is: ", len(data_dict.keys()))
print("The number of features for each person is: ", len(data_dict['METTS MARK'].keys()))

# Identifying the people with a lot of data missing to eliminate them from the dataset
low_data_ppl = []
for person in data_dict.keys():
    count = 0
    for feature in financial_features + email_features:
        if data_dict[person][feature] == "NaN":
            count += 1
    if count > 13:
        low_data_ppl.append(person)
    data_dict[person]["count"] = count

# Elimination of the people with a lot of data missing. Verification that no POI is being eliminated
for person in low_data_ppl:
    print(person, 'will be eliminated from the dataset. POI status:', data_dict[person]["poi"])
    data_dict.pop(person)

# feature_NaN = {}
# for feature in financial_features + email_features:
#     count = 0
#     for person in data_dict.keys():
#         if data_dict[person][feature] == "NaN":
#             count += 1
#     feature_NaN[feature] = count

# Identification of the people with extreme of the features to find more outliers

for feature in financial_features + email_features + ["count"]:
    key_max = max(data_dict.keys(), key=lambda k: data_dict[k][feature]
    if isinstance(data_dict[k][feature], int) else float("-inf"))
    key_min = min(data_dict.keys(), key=lambda k: data_dict[k][feature]
    if isinstance(data_dict[k][feature], int) else float("+inf"))
    max_value = data_dict[key_max][feature]
    min_value = data_dict[key_min][feature]

    print(f"{key_max} is the person with the max {feature}: {max_value} ")
    print(f"{key_min} is the person with the min {feature}: {min_value}")

"""Task 2: Remove outliers"""

data_dict.pop("TOTAL", 0)

features_list = ['poi', 'salary', 'total_payments', 'bonus', 'total_stock_value']
data = featureFormat(data_dict, features_list)

# for feature in features_list:
#     if feature is not 'poi':
#         x_points = []
#         y_points = []
#
# poi = []
# salary = []
# bonus = []
# for point in data:
#     poi.append(point[0])
#     salary.append(point[1])
#     bonus.append(point[2])
#
# matplotlib.pyplot.scatter(salary, bonus, c=poi)
# matplotlib.pyplot.xlabel("salary")
# matplotlib.pyplot.ylabel("bonus")
# matplotlib.pyplot.show()


"""Task 3: Create new feature(s)
Store to my_dataset for easy export below."""

"""Idea interaction with POI. Check how often mails from this persons are sent/received to POI"""
my_dataset = data_dict

for person in my_dataset.keys():
    if my_dataset[person]["from_messages"] != "NaN" and my_dataset[person]["to_messages"] != "NaN" and \
            my_dataset[person]["from_this_person_to_poi"] != "NaN" and \
            my_dataset[person]["from_poi_to_this_person"] != "NaN":
        my_dataset[person]["interaction_POI"] = \
            100 * (my_dataset[person]["from_this_person_to_poi"] + my_dataset[person]["from_poi_to_this_person"]) / \
            (my_dataset[person]["from_messages"] + my_dataset[person]["to_messages"])
    else:
        my_dataset[person]["interaction_POI"] = "NaN"

# Extract features and labels from dataset for local testing
features_list.append("interaction_POI")
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

"""Task 4: Try a variety of classifiers
Please name your classifier clf for easy export below.
Note that if you want to do PCA or other multi-stage operations,
you'll need to use Pipelines. For more info:
http://scikit-learn.org/stable/modules/pipeline.html"""

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

features = np.array(features)
labels = np.array(labels)

# As the labels are very unbalanced StratifiedShuffleSplit was used to separate train and test. Only 1 split was created.
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
sss.get_n_splits(features,labels)

for train_index, test_index in sss.split(features,labels):
    print("The train indexes are:", train_index)
    print("The test indexes are:", test_index)
    features_train, features_test = features[train_index], features[test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]

# Param space definition for the PCA and the different models
param_space = {
    SVC : [
        {
            'pca__n_components': [2, 5],
            'clf__kernel': [ 'sigmoid', 'poly','rbf'],
            'clf__C':  [1, 5, 10, 100, 1000],
            'clf__gamma': ['scale'],
            'clf__class_weight': ['balanced', {1: 5}, {1:10}]
        }
        ],
    DecisionTreeClassifier : [
        {
            'pca__n_components': [2, 5],
            'clf__criterion': ['gini', 'entropy'],
            'clf__max_depth': [2, 5, 10],
            'clf__class_weight': ['balanced', {1: 5}, {1:10}]
        }
    ],
    KNeighborsClassifier : [
        {
            'pca__n_components': [2, 5],
            'clf__n_neighbors': [2, 4, 6, 10],
            'clf__weights': ['distance', 'uniform'],
            'clf__algorithm': ['kd_tree', 'ball_tree', 'auto', 'brute']
        }
    ],
    RandomForestClassifier : [
        {
            'pca__n_components': [2, 5],
            'clf__n_estimators': [25, 30, 50, 100],
            'clf__criterion': ['gini', 'entropy'],
            'clf__max_depth': [3, 6, 8, 15, 20],
            'clf__class_weight': ['balanced', {1: 5}, {1:10}]
        }
    ],
}

# Models testing
print('Testing of the different models')
for Model in [SVC, DecisionTreeClassifier, KNeighborsClassifier, RandomForestClassifier]:
    print('Model: ', Model)
    t0 = time()

    pipe = Pipeline([
        ('pca', PCA()),
        ('clf', Model()),
    ])
    print(param_space[Model])
    parameters = param_space[Model]
    clf = GridSearchCV(pipe, parameters, verbose=1, scoring='recall', n_jobs=-1, cv=5)
    clf.fit(features_train, labels_train)

    labels_pred = clf.predict(features_test)
    print('The confusion matrix is:')
    print(confusion_matrix(labels_test, labels_pred))

    print('The classification report is:')
    print(classification_report(labels_test, labels_pred))
    print("done in %0.3fs" % (time() - t0))

    print('The parameters chosen are:', clf.best_params_)



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


clf = Pipeline([
    ('pca', PCA(n_components=2)),
    ('clf', RandomForestClassifier(class_weight='balanced', criterion='gini', max_depth=3, n_estimators=100))
])

clf.fit(features_train, labels_train)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
