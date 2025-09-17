# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 11:43:29 2025

@author: Rupali
"""


'''
Problem Statement: -
This dataset contains information of users in a social network. 
This social network has several business clients which can post ads on it. 
One of the clients has a car company which has just launched a luxury SUV 
for a ridiculous price. Build a Bernoulli Naïve Bayes model using this dataset 
and classify which of the users of the social network are going to purchase this luxury SUV. 
1 implies that there was a purchase and 0 implies there wasn’t a purchase.
'''
-----------------------------------------------------------------------------------------------------------------------------------------------------
##1) Business Problem
'''
1.1.	What is the business objective?
        Use the data to perdict which users will buy the SUV, so the company can show
        ads to the right people and not waste money on uninterested users.
        
1.2.	Are there any constraints?
        The model needs simple yes/no (0 or 1) data, so we had to simplify age and salary. 
        dataset has only basic details like gender, age, and salary - there is no information
        about the user interests or behaviour.
        
'''
------------------------------------------------------------------------------------------------------------------------------------------------------
##2) Business Understanding
'''
Name of Feature   |   Description            |   Type                        |   Relevance                                                       |
------------------|--------------------------|-------------------------------|-------------------------------------------------------------------|     
                  |                          |                               |                                                                   |
User ID           | Unique Identifier for    | Qualitative, Nominal          |  Irrelevant for modeling;used for identification                  |
                  | each user                |                               |                                                                   |  
------------------|--------------------------|-------------------------------|-------------------------------------------------------------------|
Gender            | Gender of the user       | Qualitative, Nominal          | Relevant: segmenting and targeting                                |  
------------------|--------------------------|-------------------------------|-------------------------------------------------------------------|
Age               | Age of the user          | Quantitative, Discrete        | Relevant:Correlates with Gender and user behavior                 |   
------------------|--------------------------|-------------------------------|-------------------------------------------------------------------|
Estimated Salary  | Estimated salary of user | Quantitative, Continuous      | Relevant: Purchasing Power and interest in high-priced products   |
------------------|--------------------------|-------------------------------|-------------------------------------------------------------------|
Purchased         | Whether the user purchased| Quantitative, Binary         | Target Variable                                                   |
------------------|---------------------------|------------------------------|-------------------------------------------------------------------|    

'''
##Import libraries
import pandas as pd                 
import numpy as np

##Load the Dataset
car = pd.read_csv("R:/Sanjivani_Assignments_2/NB_Car_Ad.csv")
car
'''
Output:
User ID  Gender  Age  EstimatedSalary  Purchased
0    15624510    Male   19            19000          0
1    15810944    Male   35            20000          0
2    15668575  Female   26            43000          0
3    15603246  Female   27            57000          0
4    15804002    Male   19            76000          0
..        ...     ...  ...              ...        ...
395  15691863  Female   46            41000          1
396  15706071    Male   51            23000          1
397  15654296  Female   50            20000          1
398  15755018    Male   36            33000          0
399  15594041  Female   49            36000          1

[400 rows x 5 columns]
'''

##Basic information
print(car.shape)
##Output: (400, 5)

##Information about the data
print(car.info())
'''
Output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 400 entries, 0 to 399
Data columns (total 5 columns):
 #   Column           Non-Null Count  Dtype 
---  ------           --------------  ----- 
 0   User ID          400 non-null    int64 
 1   Gender           400 non-null    object
 2   Age              400 non-null    int64 
 3   EstimatedSalary  400 non-null    int64 
 4   Purchased        400 non-null    int64 
dtypes: int64(4), object(1)
memory usage: 15.8+ KB
None
'''

##First five rows of the dataset
print(car.head())
'''
Output:
    User ID  Gender  Age  EstimatedSalary  Purchased
0  15624510    Male   19            19000          0
1  15810944    Male   35            20000          0
2  15668575  Female   26            43000          0
3  15603246  Female   27            57000          0
4  15804002    Male   19            76000          0
'''

print(car.columns)
'''
Output: Index(['Gender', 'Age', 'EstimatedSalary', 'Purchased'], dtype='object')
'''

## Check Missing Values
print(car.isnull().sum())
'''
Output:
User ID            0
Gender             0
Age                0
EstimatedSalary    0
Purchased          0
dtype: int64
'''
##There is no missing values in the dataset

car['Purchased'].value_counts()
'''
Output:
Purchased
0    257
1    143
Name: count, dtype: int64
'''

#############################################################
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Purchased', data=car)
plt.title("Purchased")
plt.show() 
##(0) - Most users did not purchase the SUV, means Imbalanced.

###EDA
##Gender Vs Purchase
sns.countplot(x='Gender', hue='Purchased', data=car)
plt.title("Gender Vs Purchased")
plt.show() 

###Data Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler

##Drop User ID
car.drop(['User ID'], axis=1, inplace=True)

###Gender Encoding
le = LabelEncoder()
car['Gender'] = le.fit_transform(car['Gender'])

##Feature and Target
X = car[['Gender', 'Age', 'EstimatedSalary']]
y = car['Purchased']

###Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

##Train Test Split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

###Model Training
from sklearn.naive_bayes import BernoulliNB
nb = BernoulliNB(alpha=1.0)
nb.fit(X_train, y_train)

##Predictions
y_pred_test = nb.predict(X_test)
y_pred_train = nb.predict(X_train)

###Evaluation Without Laplace Smoothing
print("Without Laplace Smoothing")
print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
##Output: Train Accuracy: 0.734375
print("Test Accuracy:", accuracy_score(y_test,  y_pred_test))
##Output: Test Accuracy: 0.6875
print("Confusion Matrix:", confusion_matrix(y_test, y_pred_test))
'''
Output:
Confusion Matrix: [[41 10]
                   [15 14]]
'''
print("Classification Report:", classification_report(y_test, y_pred_test))
'''
Output:
Classification Report:               precision    recall  f1-score   support

           0       0.73      0.80      0.77        51
           1       0.58      0.48      0.53        29

    accuracy                           0.69        80
   macro avg       0.66      0.64      0.65        80
weighted avg       0.68      0.69      0.68        80

'''

# model with Laplace Smoothing (e.g., alpha = 3)
nb_laplace = BernoulliNB(alpha=3)
nb_laplace.fit(X_train, y_train)
##Output: BernoulliNB(alpha=3)

##Predictions
y_pred_train_lp = nb_laplace.predict(X_train)
y_pred_test_lp = nb_laplace.predict(X_test)

##Evaluation With Laplace Smoothing
print("With Laplace Smoothing")
print("Train Accuracy:", accuracy_score(y_train, y_pred_train_lp))
##Output: Train Accuracy: 0.734375
print("Test Accuracy:", accuracy_score(y_test, y_pred_test_lp))
##Output: Test Accuracy: 0.6875
print("Confusion Matrix:", confusion_matrix(y_test, y_pred_test_lp))
'''
Output:
Confusion Matrix: [[41 10]
                   [15 14]]
'''
print("Classification Report:", classification_report(y_test, y_pred_test_lp))
'''
Output:
Classification Report:               precision    recall  f1-score   support

           0       0.73      0.80      0.77        51
           1       0.58      0.48      0.53        29

    accuracy                           0.69        80
   macro avg       0.66      0.64      0.65        80
weighted avg       0.68      0.69      0.68        80
'''


########Benefits and Impact of the provided solution
'''
This model helps the company show ads only to people likely
to buy the SUV, saving money and increasing sales. 
It makes marketing smarter, faster, and more effective.
'''
