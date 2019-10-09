#!/usr/bin/env python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

poi_names = "../final_project/poi_names.txt"

print("The number of people contained in the dataset is: ", len(enron_data.keys()))
print("The number of features for each person is: ", len(enron_data['METTS MARK'].keys()))

poi = []
for person_name in enron_data.keys():
    if enron_data[person_name]["poi"] == 1:
        poi.append(person_name)

for person_name in enron_data.keys():
    if person_name[0:2] == "PR":
        print(person_name)

key_name = "total_stock_value"
person_name ="PRENTICE JAMES"
print("The %s of %s is %f" % (key_name, person_name, enron_data[person_name][key_name]))

for person_name in enron_data.keys():
    if person_name[0:2] == "PR":
        print(person_name)

key_name = "from_this_person_to_poi"
person_name ="COLWELL WESLEY"
print("The %s of %s is %f" % (key_name, person_name, enron_data[person_name][key_name]))

for person_name in enron_data.keys():
    if person_name[0:2] == "SK":
        print(person_name)

key_name = "exercised_stock_options"
person_name ="SKILLING JEFFREY K"
print("The %s of %s is %f" % (key_name, person_name, enron_data[person_name][key_name]))


key_name = "total_payments"
for person_name in ["FASTOW ANDREW S", "LAY KENNETH L", "SKILLING JEFFREY K"]:
    print("The %s of %s is %f" % (key_name, person_name, enron_data[person_name][key_name]))

salary = []
key_name = "salary"
for person_name in enron_data.keys():
    if not enron_data[person_name][key_name] == "NaN":
        salary.append(person_name)

email = []
key_name = "email_address"
for person_name in enron_data.keys():
    if not enron_data[person_name][key_name] == "NaN":
        email.append(person_name)

total_payments=[]
key_name = "total_payments"
for person_name in enron_data.keys():
    if enron_data[person_name][key_name] == "NaN":
        total_payments.append(person_name)

total_payments=[]
key_name = "total_payments"
for person_name in enron_data.keys():
    if enron_data[person_name][key_name] == "NaN" and enron_data[person_name]["poi"] == 1:
        total_payments.append(person_name)

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


# the input features we want to use
# can be any key in the person-level dictionary (salary, director_fees, etc.)
feature_1 = "salary"
feature_2 = "exercised_stock_options"
poi = "poi"
features_list = [poi, feature_1, feature_2]

data = featureFormat(enron_data, features_list)
poi, finance_features = targetFeatureSplit(data)


def get_enron_list(data, person, key, key_value ):
    out = []
    for person in data.keys():
        if enron_data[person][key] == key_value:
            out.append(person)
    return out
