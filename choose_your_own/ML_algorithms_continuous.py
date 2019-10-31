#!/usr/bin/env python3

import numpy
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from class_vis import prettyPicture, output_image

def main():
    ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()

    reg = studentReg(ages_train, net_worths_train)
    print("slope: %f" % reg.coef_)
    print("intercept: %f" % reg.intercept_)

    print("Statistics on the training dataset (r-squared score): %f" % reg.score(ages_train, net_worths_train))
    print("Statistics on the testing dataset (r-squared score): %f" % reg.score(ages_test, net_worths_test))

    plt.clf()
    plt.scatter(ages_train, net_worths_train, color="b", label="train data")
    plt.scatter(ages_test, net_worths_test, color="r", label="test data")
    plt.plot(ages_test, reg.predict(ages_test), color="black")
    plt.legend(loc=2)
    plt.xlabel("ages")
    plt.ylabel("net worths")


    plt.savefig("regression.png")
 #   output_image("regression.png", "png", open("regression.png", "rb").read())


def studentReg(ages_train, net_worths_train):
    ### import the sklearn regression module
    from sklearn import linear_model

    ### create, and train your regression, ame your regression reg
    reg = linear_model.LinearRegression()
    reg.fit(ages_train, net_worths_train)

    return reg


def ageNetWorthData():
    import random

    random.seed(42)
    n_points = 200
    x = [random.randrange(0,65) for _ in range(0, n_points)]
    error = [random.randrange(-100,100)/1000 for _ in range(0, n_points)]
    net_worths = [[round(6.25*x[ii]*(1+error[ii]))] for ii in range(0, n_points)]
    ages = [[age] for age in x]
    # split into train/test sets
    split = int(0.75 * n_points)
    ages_train = ages[0:split]
    ages_test = ages[split:]
    net_worths_train = net_worths[0:split]
    net_worths_test = net_worths[split:]

    return ages_train, ages_test, net_worths_train, net_worths_test


if __name__ == '__main__':
    main()