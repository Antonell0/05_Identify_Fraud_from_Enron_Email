#!/usr/bin/env python3

import sys
import pickle

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
features_list = POI_label + financial_features + ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi'] #['poi', 'salary', 'total_payments', 'loan_advances', 'bonus']  # You will need to use more features

# Load the dictionary containing the dataset
dataset = "final_project_dataset.pkl"
with open(dataset, 'rb') as file:
    data_dict = pickle.load(file)

# Initial exploration
print("The number of people contained in the dataset is: ", len(data_dict.keys()))
print("The number of features for each person is: ", len(data_dict['METTS MARK'].keys()))

low_data_ppl = []
for person in data_dict.keys():
    count = 0
    for feature in financial_features + email_features:
        if data_dict[person][feature] == "NaN":
            count += 1
    if count > 13:
        low_data_ppl.append(person)
    data_dict[person]["count"] = count

for person in low_data_ppl:
    print(data_dict[person]["poi"])
    data_dict.pop(person)

feature_NaN = {}
for feature in financial_features + email_features:
    count = 0
    for person in data_dict.keys():
        if data_dict[person][feature] == "NaN":
            count += 1
    feature_NaN[feature] = count



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



for feature in []
poi = []
salary = []
bonus = []
for point in data:
    poi.append(point[0])
    salary.append(point[1])
    bonus.append(point[2])

matplotlib.pyplot.scatter(salary, bonus, c=poi)
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()






"""Task 3: Create new feature(s)
Store to my_dataset for easy export below."""

"""Idea interaction with POI. Check how often mails from this persons are sent/received to POI"""
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

"""Task 4: Try a varity of classifiers
Please name your classifier clf for easy export below.
Note that if you want to do PCA or other multi-stage operations,
you'll need to use Pipelines. For more info:
http://scikit-learn.org/stable/modules/pipeline.html"""

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

"""Task 5: Tune your classifier to achieve better than .3 precision and recall 
using our testing script. Check the tester.py script in the final project
folder for details on the evaluation method, especially the test_classifier
function. Because of the small size of the dataset, the script uses
stratified shuffle split cross validation. For more info: 
http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html"""

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

"""Task 6: Dump your classifier, dataset, and features_list so anyone can
check your results. You do not need to change anything below, but make sure
that the version of poi_id.py that you submit can be run on its own and
generates the necessary .pkl files for validating your results."""

dump_classifier_and_data(clf, my_dataset, features_list)



#function to extract the key with the max and min value
def extract_maxmin(dict_data, features_list):
    key_max = []
    key_min =[]
    max_value = []
    min_value = []
    for item in features
        key_max.append(max(data_dict.keys(), key=lambda k: data_dict[k][item]
                      if isinstance(data_dict[k][item], int) else float("-inf")))
        key_min.append(min(data_dict.keys(), key=lambda k: data_dict[k][item]
                      if isinstance(data_dict[k][item], int) else float("+inf")))
        max_value.append(data_dict[key_max][item])
        min_value.append(data_dict[key_min][item])
    return key_max, key_min, max_value, min_value