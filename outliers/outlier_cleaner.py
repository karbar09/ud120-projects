#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import numpy as np

    ### your code goes here
    residuals =[ x*x for x in (np.array(predictions)-np.array(net_worths)).flatten()]
    cutoff = np.percentile(residuals,90)

    temp1 = zip(ages.flatten(), net_worths.flatten(), residuals)
    temp2 = []
    for t in temp1:
        if t[2] <= cutoff:
            temp2.append(t)


    return temp1

