#!/usr/bin/env python3

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pandas as pd

financial_features= ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                     'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                     'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

email_features= ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi']

POI_label= ['poi']


"""Used features
features_list is a list of strings, each of which is a feature name.
The first feature must be "poi"."""
features_list = ['poi', 'salary', 'total_payments']  # You will need to use more features


# Load the dictionary containing the dataset
dataset = "final_project_dataset.pkl"
with open(dataset, 'rb') as file:
    data_dict = pickle.load(file)

#Initial exploration
print("The number of people contained in the dataset is: ", len(data_dict.keys()))
print("The number of features for each person is: ", len(data_dict['METTS MARK'].keys()))


key_max = max(data_dict, key=lambda k: data_dict[k]['total_payments'] if isinstance(data_dict[k]['total_payments'],float) else float("-inf"))
key_min = min(data_dict, key=lambda k: data_dict[k]['total_payments'] if isinstance(data_dict[k]['total_payments'],float) else float("+inf"))



for feature in financial_features, email_features:
    for person_name in data_dict.keys():



df = pd.DataFrame.from_dict(data_dict, orient = 'index')
df[['salary']] = df[['salary']].apply(pd.to_numeric)
df[['deferral_payments']] = df[['deferral_payments']].apply(pd.to_numeric, errors='coerce')
df[['total_payments']] = df[['total_payments']].apply(pd.to_numeric, errors='coerce')
df[['restricted_stock_deferred']] = df[['restricted_stock_deferred']].apply(pd.to_numeric, errors='coerce')
df[['exercised_stock_options']] = df[['exercised_stock_options']].apply(pd.to_numeric, errors='coerce')
df[['long_term_incentive']] = df[['long_term_incentive']].apply(pd.to_numeric, errors='coerce')
df[['bonus']] = df[['salary']].apply(pd.to_numeric, errors='coerce')
df[['total_stock_value']] = df[['total_stock_value']].apply(pd.to_numeric, errors='coerce')


df.describe()
df.info()

ax1 = df.plot.scatter(x='salary',y='total_payments',c='deferral_payments')

missing_data = []
for key_name in features_list:
    for person_name in data_dict.keys():
        if data_dict[person_name][key_name] == "NaN":
            missing_data.append(person_name + "_" + key_name)

print(missing_data)

"""Task 2: Remove outliers



features_list

Task 3: Create new feature(s)
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
