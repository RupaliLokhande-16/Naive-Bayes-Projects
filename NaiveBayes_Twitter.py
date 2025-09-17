# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 12:34:04 2025

@author: Rupali
"""

'''
Problem Statement: -
In this case study, you have been given Twitter data collected from 
an anonymous twitter handle. With the help of a Naïve Bayes model, 
predict if a given tweet about a real disaster is real or fake.
1 = real tweet and 0 = fake tweet
'''
-----------------------------------------------------------------------------------------
##1) Business Problem
'''
1.1. What is the business objective?
        Use the data to predict which tweets are about a REAL disaster, so platforms and
        responders can prioritize critical information and avoid noise.

1.2. Are there any constraints?
        Tweets are short, noisy, and unstructured (URLs, hashtags, emojis, abbreviations).
        Some metadata (keyword, location) is missing or unreliable.
        Real-time inference may be required for streaming data.
'''
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##2) Business Understanding
'''
Name of Feature   |   Description                           |   Type                        |   Relevance
------------------|-----------------------------------------|-------------------------------|-------------------------------------------------------------------
id                | Unique Identifier for each tweet        | Qualitative, Nominal          | Irrelevant for modeling; used only for identification
------------------|-----------------------------------------|-------------------------------|-------------------------------------------------------------------
keyword           | Parsed keyword from tweet (if present)  | Qualitative, Nominal          | Relevant: hints at disaster topic (e.g., "earthquake", "flood")
------------------|-----------------------------------------|-------------------------------|-------------------------------------------------------------------
location          | User-provided location text (optional)  | Qualitative, Nominal          | Partially relevant: often missing/noisy but may add context
------------------|-----------------------------------------|-------------------------------|-------------------------------------------------------------------
text              | The raw tweet content                   | Qualitative, Text             | Highly relevant: main evidence of disaster/non-disaster
------------------|-----------------------------------------|-------------------------------|-------------------------------------------------------------------
target            | 1 = real disaster, 0 = not a disaster   | Quantitative, Binary          | Target Variable
'''

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##import libraries
import pandas as pd
import numpy as np

##Load the dataset
twitter = pd.read_csv("R:/Sanjivani_Assignments_2/Disaster_tweets.csv")
twitter


#########Basic Information
print(twitter.shape)
##Output: (7613, 5)

##Information of the data
print(twitter.info())
'''
Output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7613 entries, 0 to 7612
Data columns (total 5 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   id        7613 non-null   int64 
 1   keyword   7552 non-null   object
 2   location  5080 non-null   object
 3   text      7613 non-null   object
 4   target    7613 non-null   int64 
dtypes: int64(2), object(3)
memory usage: 297.5+ KB
None
'''

###Print first five rows of the dataset
print(twitter.head())
'''
Output:
 id keyword  ...                                               text target
0   1     NaN  ...  Our Deeds are the Reason of this #earthquake M...      1
1   4     NaN  ...             Forest fire near La Ronge Sask. Canada      1
2   5     NaN  ...  All residents asked to 'shelter in place' are ...      1
3   6     NaN  ...  13,000 people receive #wildfires evacuation or...      1
4   7     NaN  ...  Just got sent this photo from Ruby #Alaska as ...      1

[5 rows x 5 columns]
'''

##Columns of the dataset
print(twitter.columns)
##Output: Index(['id', 'keyword', 'location', 'text', 'target'], dtype='object')

###Checking for the missing values
print(twitter.isnull().sum())
'''
Output:
id             0
keyword       61
location    2533
text           0
target         0
dtype: int64
'''

####Target distribution 
print(twitter['target'].value_counts())
'''
Output:
target
0    4342
1    3271
Name: count, dtype: int64
'''

#####################################################################################################################
import seaborn as sns
import matplotlib.pyplot as plt

###Target distribution plot
sns.countplot(x='target', data=twitter)
plt.title("Target (0=Not Disaster, 1=Disaster)")
plt.show()
### 0s are more than 1s


############EDA
##keyword vs target
twitter['has_keyword'] = twitter['keyword'].notna().astype(int)
sns.countplot(x='has_keyword', hue='target', data=twitter)
plt.title("Keyword Vs Target")
plt.xlabel("has_keyword (1=present)")
plt.show()

------------------------------------------------------------------------------------------------------------------------
##Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

###Fill the missing values
twitter['keyword'] = twitter['keyword'].fillna('')
twitter['location'] = twitter['location'].fillna('')

##Combining keyword with text
twitter['combined_text'] = (twitter['keyword'].str.replace('%20',' ', regex=False) + ' ' + twitter['text']).str.strip()

###Feature and Target
X_test = twitter['combined_text']
y = twitter['target']

##Vectirizing the text
vectorizer = CountVectorizer(binary=True, stop_words='english')
X = vectorizer.fit_transform(X_test)

###Now Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

##Now training the Model
from sklearn.naive_bayes import BernoulliNB
nb = BernoulliNB(alpha=1.0)
nb.fit(X_train, y_train)
##Output: BernoulliNB()

###Predictions
y_pred_test = nb.predict(X_test)
y_pred_train = nb.predict(X_train)

### Evaluation Without Laplace Smoothing
print("Without Laplace Smoothing")
print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
##Output: Train Accuracy: 0.899671592775041

print("Test Accuracy:", accuracy_score(y_test,  y_pred_test))
##Output: Test Accuracy: 0.8115561391989494

print("Confusion Matrix:", confusion_matrix(y_test, y_pred_test))
'''
Output: 
Confusion Matrix: [[803  66]
                   [221 433]]
'''

print("Classification Report:", classification_report(y_test, y_pred_test))
'''
Output:
Classification Report:               precision    recall  f1-score   support

           0       0.78      0.92      0.85       869
           1       0.87      0.66      0.75       654

    accuracy                           0.81      1523
   macro avg       0.83      0.79      0.80      1523
weighted avg       0.82      0.81      0.81      1523
'''
----------------------------------------------------------------------------------------------------------------------

##Model with Laplace Smoothing
nb_laplace = BernoulliNB(alpha=3)
nb_laplace.fit(X_train, y_train)
##Output: BernoulliNB(alpha=3)

## Predictions
y_pred_train_lp = nb_laplace.predict(X_train)
y_pred_test_lp = nb_laplace.predict(X_test)

## Evaluation With Laplace Smoothing
print("With Laplace Smoothing (alpha=3)")
print("Train Accuracy:", accuracy_score(y_train, y_pred_train_lp))
##Output: Train Accuracy: 0.7559934318555008

print("Test Accuracy:", accuracy_score(y_test, y_pred_test_lp))
##output: Test Accuracy: 0.7248850952068286

print("Confusion Matrix:", confusion_matrix(y_test, y_pred_test_lp))
'''
Output:
Confusion Matrix: [[862   7]
                   [412 242]]
'''

print("Classification Report:", classification_report(y_test, y_pred_test_lp))
'''
Output:
Classification Report:               precision    recall  f1-score   support

           0       0.68      0.99      0.80       869
           1       0.97      0.37      0.54       654

    accuracy                           0.72      1523
   macro avg       0.82      0.68      0.67      1523
weighted avg       0.80      0.72      0.69      1523
'''


######## Benefits and Impact of the provided solution
'''
This Model helps to find real disaster tweets quickly, 
saves time, reduces confusion, and supports faster emergency response.
'''
