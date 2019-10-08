#!/usr/bin/env python3

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()

"""the training data (features_train, labels_train) have both "fast" and "slow"
points mixed together--separate them so we can give them different colors
in the scatterplot and identify them visually"""

grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]


# initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()

""" your code here!  name your classifier object clf if you want the 
visualization code (prettyPicture) to show you the decision boundary"""

algo_choice = ["GradientBoosting", "AdaBoost", "Random Forest", "KNN", "DecisionTree", "SVM", "GaussianNB"]

for algorithm in algo_choice:
    print(algorithm)
    if algorithm == "GaussianNB":
        print("The selected algorithm is ", algorithm)
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
        clf.fit(features_train, labels_train)
        labels_pred = clf.predict(features_test)

    if algorithm == "SVM":
        print("The selected algorithm is ", algorithm)
        from sklearn.svm import SVC
        clf = SVC(kernel="linear")
        clf.fit(features_train, labels_train)
        labels_pred = clf.predict(features_test)

    if algorithm == "DecisionTree":
        print("The selected algorithm is ", algorithm)
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier()
        clf.fit(features_train, labels_train)
        labels_pred = clf.predict(features_test)

    if algorithm == "KNN":
        print("The selected algorithm is ", algorithm)
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(features_train, labels_train)
        labels_pred = clf.predict(features_test)

    if algorithm == "Random Forest":
        print("The selected algorithm is ", algorithm)
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, max_features="auto",random_state=0)
        clf.fit(features_train, labels_train)
        labels_pred = clf.predict(features_test)

    if algorithm == "AdaBoost":
        print("The selected algorithm is ", algorithm)
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier(n_estimators=100)
        clf.fit(features_train, labels_train)
        labels_pred = clf.predict(features_test)

    if algorithm == "GradientBoosting":
        print("The selected algorithm is ", algorithm)
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=100)
        clf.fit(features_train, labels_train)
        labels_pred = clf.predict(features_test)

    fig_name = algorithm + "png"

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(labels_test, labels_pred, normalize=True, sample_weight=None)
    print(accuracy)

    try:
        prettyPicture(clf, features_test, labels_test, fig_name)
    except NameError:
        pass
