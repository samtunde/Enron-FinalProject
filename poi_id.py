
# coding: utf-8

# In[35]:

# %load poi_id.py
#!/usr/bin/python

import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat
from feature_format import targetFeatureSplit
import tester
from tester import dump_classifier_and_data


#

from sklearn.metrics import accuracy_score
#from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                'salary',
                'bonus', 
                'long_term_incentive', 
                'deferred_income', 
                'deferral_payments',
                'loan_advances', 
                'other',
                'expenses', 
                'director_fees',
                'total_payments',
                'exercised_stock_options',
                'restricted_stock',
                'restricted_stock_deferred',
                'total_stock_value',
                'to_messages',
                'from_messages',
                'from_this_person_to_poi',
                'from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Transform data from dictionary to the Pandas DataFrame
df = pd.DataFrame.from_dict(data_dict, orient = 'index')
#Order columns in DataFrame, exclude email column
df = df[features_list]
df = df.replace('NaN', np.nan)
df.info()


# In[38]:

df.count().sort_values()


#'email_address' variable
#df = df.drop(["email_address"], axis=1)


print "Total number of NaN values in the dataset: ", df.isnull().sum().sum()


# Replace NaN in financial features with 0
df.ix[:,:15] = df.ix[:,:15].fillna(0)


print "Total number of NaN values in the dataset: ", df.isnull().sum().sum()


### Task 2a: Cleaning Data file

#split of POI and non-POI in the dataset
poi_non_poi = df.poi.value_counts()
poi_non_poi.index=['non-POI', 'POI']

# Replace NaN in financial features with 0
df.ix[:,:15] = df.ix[:,:15].fillna(0)

email_features = ['to_messages', 'from_messages', 'from_this_person_to_poi', 'from_poi_to_this_person']

imp = Imputer(missing_values='NaN', strategy='median', axis=0)

#impute missing values of email features 
df.loc[df[df.poi == 1].index,email_features] = imp.fit_transform(df[email_features][df.poi == 1])
df.loc[df[df.poi == 0].index,email_features] = imp.fit_transform(df[email_features][df.poi == 0])

## Review financial data accuracy
#check data: summing payments features and compare with total_payments
payments = ['salary',
            'bonus', 
            'long_term_incentive', 
            'deferred_income', 
            'deferral_payments',
            'loan_advances', 
            'other',
            'expenses', 
            'director_fees']
df[df[payments].sum(axis='columns') != df.total_payments]

stock_value = ['exercised_stock_options',
                'restricted_stock',
                'restricted_stock_deferred']
df[df[stock_value].sum(axis='columns') != df.total_stock_value]

## Update financial data for BHATNAGAR SANJAY and BELFER ROBERT
df.ix['BELFER ROBERT','total_payments'] = 3285
df.ix['BELFER ROBERT','deferral_payments'] = 0
df.ix['BELFER ROBERT','restricted_stock'] = 44093
df.ix['BELFER ROBERT','restricted_stock_deferred'] = -44093
df.ix['BELFER ROBERT','total_stock_value'] = 0
df.ix['BELFER ROBERT','director_fees'] = 102500
df.ix['BELFER ROBERT','deferred_income'] = -102500
df.ix['BELFER ROBERT','exercised_stock_options'] = 0
df.ix['BELFER ROBERT','expenses'] = 3285
df.ix['BELFER ROBERT',]
df.ix['BHATNAGAR SANJAY','expenses'] = 137864
df.ix['BHATNAGAR SANJAY','total_payments'] = 137864
df.ix['BHATNAGAR SANJAY','exercised_stock_options'] = 1.54563e+07
df.ix['BHATNAGAR SANJAY','restricted_stock'] = 2.60449e+06
df.ix['BHATNAGAR SANJAY','restricted_stock_deferred'] = -2.60449e+06
df.ix['BHATNAGAR SANJAY','other'] = 0
df.ix['BHATNAGAR SANJAY','director_fees'] = 0
df.ix['BHATNAGAR SANJAY','total_stock_value'] = 1.54563e+07
df.ix['BHATNAGAR SANJAY']
df[df[payments].sum(axis='columns') != df.total_payments]


#df = df[features_list]
#df.head(10)
print "Total number of NaN values in the dataset: ", df.isnull().sum().sum()


# # Outlier Investigation
# Now the data has been cleaned from missing values and typos I would like to discover the outliers. Descriptive statistics can be used to determine outliers of the distibution as the values which are higher than Q2 + 1.5IQR or less than Q2 - 1.5IQR, where Q2 median of the distribution, IQR - interquartile range. I'm going to calculate the sum of outlier variables for each person and sort them descending.


### Task 2b: Remove outliers
#Visualize outliers
import matplotlib.pyplot as plt
from feature_format import featureFormat, targetFeatureSplit

features = ["salary", "bonus"]

## remove the outlier key = TOTAL
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus, c=None)

plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.show()

## Using descriptive statistics to determine the outliers in this distribution of data
outliers = df.quantile(.5) + 1.5 * (df.quantile(.75)-df.quantile(.25))
pd.DataFrame((df[1:] > outliers[1:]).sum(axis = 1), columns = ['# of outliers']).    sort_values('# of outliers',  ascending = [0]).head(12)
    
    
# Remove Total outlier from the data set
df = df.drop(['TOTAL'],0)


# # Outlier Review:
# The first value is 'TOTAL' which is the total value of financial payments from the FindLaw data. As it's doesn't make any sence for our solution, I'm going to exclude it from the data set.
# 
# Kenneth Lay and Jeffrey Skilling are very well known persons from ENRON scandal. They will be kept in the dataset as they represent anomalies but not the outliers.
# 
# Mark Frevert and Lawrence Whalley are high level managers at Enron who could represent great examples for modeling.
# 


# In[46]:

### Task 3: Create new feature(s)
# Create new feature: fraction of person's email to POI to all sent messages
df['fraction_to_poi'] = df['from_this_person_to_poi']/df['from_messages']

# Clean all 'inf' values which we got if the person's from_messages = 0
df = df.replace('inf', 0)


# # Create New Feature
# I reviewed multiple resource allocation (POI/NON_POI) of the financial data and very little insight to the data. An email feature was created to check the fraction of emails, sent to POI, to all sent emails; emails, received from POI, to all received emails. The new feature name is "fraction_to_poi".

### create new features
### new features are: fraction_to_poi_email,fraction_from_poi_email
#from IPython.display import Image
#import matplotlib.pyplot as plt
def dict_to_list(key,normalizer):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

### create two lists of new features
fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")

### insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"]=fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"]=fraction_to_poi_email[count]
    count +=1

    
#features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email"]    
    ### store to my_dataset for easy export below
my_dataset = data_dict




# # Feature Selection
# In order to find the most effective features for classification, feature selection using “Decision Tree” was deployed to rank the features. Resulting in a number of features with non-null feature importance, sorted by importance. Note Decision tree doesn't require me any feature scaling. According to feature_importances attribute of the classifier, just created fraction_to_poi feature has the highest importance for the model. It is important to note the number of features used for the model can cause varied results.

from time import time


## features_list = ["poi", "salary", "bonus", "fraction_from_poi_email", "fraction_to_poi_email",
#                 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred',
#                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
 #                'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']

data = featureFormat(my_dataset, features_list)

### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)

### split data into training and testing datasets
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, 
                                                                labels, test_size=0.1, random_state=42)
t0 = time()

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
pred= clf.predict(features_test)
print 'accuracy', score

print "Decision tree algorithm time:", round(time()-t0, 3), "s"

importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
print 'Feature Ranking: '
for i in range(16):
    print "{} {} ({})".format(i+1,features_list[i+1],importances[indices[i]])


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


#Decision Tree Classifier with standard parametres 
clf = DecisionTreeClassifier(random_state = 75)
my_dataset = df[features_list].to_dict(orient = 'index')
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main() 


#Random Forest with standard parameters
clf = RandomForestClassifier(random_state = 75)
clf.fit(df.ix[:,1:], np.ravel(df.ix[:,:1]))

# selecting the features with non null importance, sorting and creating features_list for the model
features_importance = []
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0:
        features_importance.append([df.columns[i+1], clf.feature_importances_[i]])
features_importance.sort(key=lambda x: x[1], reverse = True)
features_list = [x[0] for x in features_importance]
features_list.insert(0, 'poi')

# number of features for best result was found iteratively
features_list2 = features_list[:11]
my_dataset = df[features_list2].to_dict(orient = 'index')
tester.dump_classifier_and_data(clf, my_dataset, features_list2)
tester.main()

# GaussianNB with feature standartization, selection, PCA
from sklearn.feature_selection import SelectKBest, f_classif
from tester import dump_classifier_and_data

clf = GaussianNB()

# data set standartization
scaler = StandardScaler()
df_norm = df[features_list]
df_norm = scaler.fit_transform(df_norm.ix[:,1:])

# feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
features_list2 = ['poi']+range(3)
my_dataset = pd.DataFrame(SelectKBest(f_classif, k=8).fit_transform(df_norm, df.poi), index = df.index)

#PCA
pca = PCA(n_components=3)
my_dataset2 = pd.DataFrame(pca.fit_transform(my_dataset),  index=df.index)
my_dataset2.insert(0, "poi", df.poi)
my_dataset2 = my_dataset2.to_dict(orient = 'index')  

dump_classifier_and_data(clf, my_dataset2, features_list2)
tester.main()


pd.DataFrame([[0.8874, 0.5761, 0.5885, 0.5822],
              [0.8978, 0.7032, 0.4040, 0.5132],
              [0.8613, 0.4773, 0.4250, 0.4496]],
             columns = ['Accuracy','Precision', 'Recall', 'F1'], 
             index = ['Decision Tree Classifier', 'Random Forest', 'Gaussian Naive Bayes'])


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


clf = DecisionTreeClassifier(criterion = 'entropy', 
                             min_samples_split = 19,
                             random_state = 60,
                             min_samples_leaf=6, 
                             max_depth = 3,
                            class_weight=None)

clf.fit(df.ix[:,1:], df.poi)

# show the features with non null importance, sorted and create features_list of features for the model
features_importance = []
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0:
        features_importance.append([df.columns[i+1], clf.feature_importances_[i]])
features_importance.sort(key=lambda x: x[1], reverse = True)

features_list = [x[0] for x in features_importance]
features_list.insert(0, 'poi')

my_dataset = df[features_list].to_dict(orient = 'index')
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()


features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email", "to_messages"]

### try Naive Bayes for prediction
t0 = time()

clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
print accuracy

print "NB algorithm time:", round(time()-t0, 3), "s"

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label
features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email", "to_messages"]


### store to my_dataset for easy export below
my_dataset = data_dict


### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)


### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)


### machine learning goes here!
### please name your classifier clf for easy export below

### deploying feature selection
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

### use KFold for split and validate algorithm
from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]

t0 = time()

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
print 'accuracy before tuning ', score

print "Decision tree algorithm time:", round(time()-t0, 3), "s"


### use manual tuning parameters
t0 = time()
clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
        max_features=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None,
        min_samples_leaf=1, min_samples_split=17,
        min_weight_fraction_leaf=0.0, presort=False, random_state=None,
        splitter='best')
clf = clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
print("done in %0.3fs" % (time() - t0))

acc=accuracy_score(labels_test, pred)

print "Validating algorithm:"
print "accuracy after tuning = ", acc

# function for calculation ratio of true positives
# out of all positives (true + false)
print 'precision = ', precision_score(labels_test,pred)

# function for calculation ratio of true positives
# out of true positives and false negatives
print 'recall = ', recall_score(labels_test,pred)


### dump your classifier, dataset and features_list so
### anyone can run/check your results
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )

