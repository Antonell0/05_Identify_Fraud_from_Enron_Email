#!/usr/bin/env python3

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


# read in data dictionary, convert to numpy array
datafile = "../final_project/final_project_dataset.pkl"
with open(datafile, 'rb') as file:
    data_dict = pickle.load(file)
features = ["salary", "bonus"]

key_name = "bonus"
for person_name in ["LAY KENNETH L"]:
    print("The %s of %s is %f" % (key_name, person_name, data_dict[person_name][key_name]))

salary = []
key_name = "salary"
key_value0 = 0.
for person_name in data_dict.keys():
    key_value = data_dict[person_name][key_name]
    if not key_value == "NaN":
        if key_value > key_value0:
            max = person_name
            key_value0 = key_value
print(max)
data_dict.pop("TOTAL", 0)

salary = []
key_name = "salary"
key_value0 = 0.
for person_name in data_dict.keys():
    key_value1 = data_dict[person_name][features[0]]
    key_value2 = data_dict[person_name][features[1]]
    if not (key_value1 == "NaN" or key_value1 == "NaN"):
        if key_value1 > 1000000 and key_value2 > 5000000:
            print("The %s of %s is %f and the %s is %f" % (features[0], person_name, key_value1, features[1], key_value2))

data = featureFormat(data_dict, features)


# your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus, color = 'blue')

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


