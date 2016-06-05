#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop("TOTAL")
data = featureFormat(data_dict, features)


for key in data_dict:
    if float(data_dict[key]["salary"]) > 1000000 and float(data_dict[key]["bonus"]) > 5000000:
        print key

### your code below
for point in data:
    salary = point[0]
    bonus = point[1]

    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


