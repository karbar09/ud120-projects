#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".'loan_advances','deferred_income','expenses']

features_list= ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person', 'from_messages', #'email_address',
                 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

"""
    Creates variable v3, which is the ratio of variables v1 to v2. v3 is NaN if v1 or v2 is NaN
    assumes v1 and v3 are numeric
    assumes d_dict has the same structure as data_dict
    returns a modified d_dict
"""
def create_ratio_var(d_dict,v1,v2,v3):
    for key,value in d_dict.iteritems():
        v1_var = value[v1]
        v2_var = value[v2]
        if v1_var != "NaN" and v2_var != "NaN":
            v3_var = (1.0*v1_var)/v2_var
        else:
            v3_var = "NaN"
        d_dict[key][v3] = v3_var
    return d_dict


#### Feature 1: ratio of stock that is exercised
data_dict= create_ratio_var(data_dict,"exercised_stock_options","total_stock_value","total_stock_exercised_ratio")
#### Feature 2: ratio of to_emails that are from_this_person_to_poi
data_dict= create_ratio_var(data_dict,"from_this_person_to_poi","from_messages","from_messages_to_poi_ratio")
#### Feature 3: ratio of from_emails that are from_poi_to_this_person
data_dict= create_ratio_var(data_dict,"from_poi_to_this_person","to_messages","to_messages_from_poi_ratio")

features_list.append('total_stock_exercised_ratio')
features_list.append('from_messages_to_poi_ratio')
features_list.append('to_messages_from_poi_ratio')


#Code to generate some summary statistics on the data
"""
import pandas as pd
df = pd.DataFrame(data_dict).transpose()
for f in features_list:
    d = df[f].describe()
    print f,d['count'],d['unique'],d['top'],d['freq']
#print df.describe()
print len(df)
print len(features_list)
"""


my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

select_k_best = SelectKBest()

combined_features = FeatureUnion([("select_k_best", select_k_best)])
dt = DecisionTreeClassifier(random_state=21)

#lg = LogisticRegression(random_state=21,solver='liblinear')

pipeline = Pipeline([("features", combined_features),
                     #('clf',lg)])
                     ("clf", dt)])

param_grid = dict(features__select_k_best__k=[3,5,10,15,'all'],
                  #clf__penalty=['l1','l2'],
                  #clf__C = [0.01,0.1,1,10],
                  #clf__fit_intercept=[True,False])
                  clf__splitter=["best","random"],
                  clf__criterion=["gini","entropy"],
                  clf__max_features=["auto","sqrt","log2",None]
                   )


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import StratifiedShuffleSplit

sss_labels = StratifiedShuffleSplit(labels,100,random_state=41)
grid = GridSearchCV(pipeline, param_grid=param_grid,cv=sss_labels,scoring="f1")

grid.fit(features,labels)
#print grid.best_estimator_
#print grid.best_score_
#print grid.best_params_
clf = grid.best_estimator_

"""
#Print Select K Best Scores
k = SelectKBest(k=1)
k.fit(features,labels)
feature_scores = sorted(zip(features_list[1:],k.scores_,k.pvalues_),  key=lambda tup:tup[1])
for f in reversed(feature_scores):
    print f[0],f[1],f[2]

#Fit Decision Tree and print feature importances
dt = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=21, splitter='best')
dt.fit(features,labels)
print " "
feature_scores = sorted(zip(features_list[1:],dt.feature_importances_,k.scores_,k.pvalues_),  key=lambda tup:tup[1])
for f in reversed(feature_scores):
    print f[0],f[1],f[2]
print "Max Features: " + str(dt.max_features_)
clf = dt
"""

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
