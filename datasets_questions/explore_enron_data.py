#!/usr/bin/python

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

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


qsalary_count = 0
email_count = 0
payment_count = 0
for key in enron_data:
	if enron_data[key]["salary"] != 'NaN':
		qsalary_count += 1
	if enron_data[key]["email_address"] != 'NaN':
		email_count += 1
	if enron_data[key]["poi"] == False and enron_data[key]["total_payments"] == 'NaN':
		payment_count += 1

print qsalary_count
print email_count 
print payment_count
print len(enron_data)

