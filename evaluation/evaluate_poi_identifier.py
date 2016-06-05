#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### your code goes here

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


print len(labels_test)
dt = DecisionTreeClassifier()
dt.fit(features_train,labels_train)
print dt.score(features_test,labels_test)
labels_pred = dt.predict(features_test)

p=0
tp = 0
for i,j in zip(labels_pred,labels_test):
    if i==1.0:
        p+=1
    if i==1.0 and j==1.0:
        tp +=1
print p,tp

from sklearn.metrics import precision_score,recall_score

print precision_score(labels_test,labels_pred)
print recall_score(labels_test,labels_pred)