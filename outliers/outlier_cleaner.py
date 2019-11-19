#!/usr/bin/env python3


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import numpy

    cleaned_data = []

    # your code goes here
    error = numpy.subtract(predictions, net_worths)
    index = numpy.squeeze(abs(error)).argsort()
    index = index[0:int(len(index) * 0.9)]

    for ii in index:
        cleaned_data.append((ages[ii], net_worths[ii], error[ii]))

    return cleaned_data

