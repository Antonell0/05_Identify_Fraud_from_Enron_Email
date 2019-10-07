#!/usr/bin/env python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

features_train = features_train[:round(len(features_train)/50)]
labels_train = labels_train[:round(len(labels_train)/50)]

from sklearn.svm import SVC

# # SVM parameters
# kernel = "linear" # more details in: https://scikit-learn.org/stable/modules/svm.html#svm-kernels
#
from sklearn.model_selection import GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = SVC(gamma="scale")
clf = GridSearchCV(svc, parameters, cv=5)

tf0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-tf0, 3), "s")

print(sorted(clf.cv_results_.keys()))



# tf0 = time()
# clf = SVC(kernel="linear")
# clf.fit(features_train, labels_train)
# print("training time:", round(time()-tf0, 3), "s")
#
tp0 = time()
labels_pred = clf.predict(features_test)
print("prediction time:", round(time()-tp0, 3), "s")

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, labels_pred)
print(accuracy)





