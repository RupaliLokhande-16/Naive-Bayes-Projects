# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 20:27:31 2025

@author: Rupali
"""

'''
Problem Statement:
Prepare a classification model using the Naive Bayes algorithm for the salary dataset. 
Train and test datasets are given separately. 
Use both for model building.
'''

------------------------------------------------------------------------------------------------------------------------------------------
## 1) Business Problem
'''
1.1. What is the business objective?
     Predict which individuals have a salary greater than 50K so that organizations
     can identify high-income groups for targeted campaigns or analysis.

1.2. Are there any constraints?
     The dataset contains both categorical and numerical features.
     The model should handle mixed types of data after preprocessing.
'''
     
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
## 2) Business Understanding
'''
Name of Feature   |   Description                               |   Type                        |   Relevance
------------------|---------------------------------------------|-------------------------------|-----------------------------------------------------
age               | Age of the individual                       | Quantitative, Continuous      | Relevant - impacts income level
workclass         | Employment type                             | Qualitative, Nominal          | Relevant - affects earning capacity
education         | Highest education qualification             | Qualitative, Ordinal          | Relevant - correlates with salary
educationno       | Numerical encoding of education level       | Quantitative, Discrete        | Relevant - direct mapping to education
maritalstatus     | Marital status                              | Qualitative, Nominal          | Relevant - linked with stability & earnings
occupation        | Type of occupation                          | Qualitative, Nominal          | Relevant - determines earning capacity
relationship      | Relationship status                         | Qualitative, Nominal          | Relevant - socioeconomic indicator
race              | Race of the individual                      | Qualitative, Nominal          | Relevant - potential correlation
sex               | Gender                                      | Qualitative, Nominal          | Relevant - may affect earning trends
capitalgain       | Capital gain in the past year               | Quantitative, Continuous      | Relevant - direct effect on income
capitalloss       | Capital loss in the past year               | Quantitative, Continuous      | Relevant - indicates loss in investments
hoursperweek      | Hours worked per week                       | Quantitative, Continuous      | Relevant - more hours = higher income potential
native            | Country of origin                           | Qualitative, Nominal          | Relevant - regional income differences
Salary            | Target Variable (>50K or <=50K)             | Qualitative, Binary           | Target variable
'''

-------------------------------------------------------------------------------------------------------------------------------------------------------------------
Classification Model on Train Data
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

##Import libraries
import pandas as pd
import numpy as np

# 3) Load Training Data
data1 = pd.read_csv("R:/Sanjivani_Assignments_2/SalaryData_Train.csv")

####Shape of the data
print(data1.shape)
##Output: (30161, 14)

##Information about the data
print(data1.info())
'''
Output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 30161 entries, 0 to 30160
Data columns (total 14 columns):
 #   Column         Non-Null Count  Dtype 
---  ------         --------------  ----- 
 0   age            30161 non-null  int64 
 1   workclass      30161 non-null  object
 2   education      30161 non-null  object
 3   educationno    30161 non-null  int64 
 4   maritalstatus  30161 non-null  object
 5   occupation     30161 non-null  object
 6   relationship   30161 non-null  object
 7   race           30161 non-null  object
 8   sex            30161 non-null  object
 9   capitalgain    30161 non-null  int64 
 10  capitalloss    30161 non-null  int64 
 11  hoursperweek   30161 non-null  int64 
 12  native         30161 non-null  object
 13  Salary         30161 non-null  object
dtypes: int64(5), object(9)
memory usage: 3.2+ MB
None
'''

##First five rows of the dataset
print(data1.head())
'''
Output:
age          workclass   education  ...  hoursperweek          native  Salary
0   39          State-gov   Bachelors  ...            40   United-States   <=50K
1   50   Self-emp-not-inc   Bachelors  ...            13   United-States   <=50K
2   38            Private     HS-grad  ...            40   United-States   <=50K
3   53            Private        11th  ...            40   United-States   <=50K
4   28            Private   Bachelors  ...            40            Cuba   <=50K

[5 rows x 14 columns]
'''


## Checking Missing Values
print(data1.isnull().sum())
'''
Output:
age              0
workclass        0
education        0
educationno      0
maritalstatus    0
occupation       0
relationship     0
race             0
sex              0
capitalgain      0
capitalloss      0
hoursperweek     0
native           0
Salary           0
dtype: int64
'''
##There are no missing values

print(data1.columns)
'''
Output:
Index(['age', 'workclass', 'education', 'educationno', 'maritalstatus',
       'occupation', 'relationship', 'race', 'sex', 'capitalgain',
       'capitalloss', 'hoursperweek', 'native', 'Salary'],
      dtype='object')
'''

# 4) Separate Features and Target
X_train = data1.drop("Salary", axis=1)
y_train = data1["Salary"]

# 5) Preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

categorical_cols = ["workclass", "education", "maritalstatus", "occupation",
                    "relationship", "race", "sex", "native"]
numeric_cols = ["age", "educationno", "capitalgain", "capitalloss", "hoursperweek"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

##########################################################################################################
# 5) Model Pipeline
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB

pipeline = Pipeline([
    ("prep", preprocess),
    ("nb", GaussianNB())
])

######################################################################################################################
# 6) Train Model
pipeline.fit(X_train, y_train)

# 8) Predictions and Evaluation on Training Data
y_pred_train = pipeline.predict(X_train)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


print("\nTrain Accuracy:", accuracy_score(y_train, y_pred_train))
##Output: Train Accuracy: 0.5864858592221743

print("\nConfusion Matrix (Train):\n", confusion_matrix(y_train, y_pred_train))
'''
Output:
Confusion Matrix (Train):
 [[10648 12005]
 [  467  7041]]
 '''
 
print("\nClassification Report (Train):\n", classification_report(y_train, y_pred_train))
'''
Output:

               precision    recall  f1-score   support

       <=50K       0.96      0.47      0.63     22653
        >50K       0.37      0.94      0.53      7508

    accuracy                           0.59     30161
   macro avg       0.66      0.70      0.58     30161
weighted avg       0.81      0.59      0.61     30161
'''

----------------------------------------------------------------------------------------------------------------------------
Classification Model on Test Dataset
----------------------------------------------------------------------------------------------------------------------------
##Import libraries
import pandas as pd
import numpy as np

# 3) Load Test Data
data2 = pd.read_csv("R:/Sanjivani_Assignments_2/SalaryData_Test.csv")

####Shape of the data
print(data2.shape)
##Output: (15060, 14)

##Information about the data
print(data2.info())
'''
Output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 15060 entries, 0 to 15059
Data columns (total 14 columns):
 #   Column         Non-Null Count  Dtype 
---  ------         --------------  ----- 
 0   age            15060 non-null  int64 
 1   workclass      15060 non-null  object
 2   education      15060 non-null  object
 3   educationno    15060 non-null  int64 
 4   maritalstatus  15060 non-null  object
 5   occupation     15060 non-null  object
 6   relationship   15060 non-null  object
 7   race           15060 non-null  object
 8   sex            15060 non-null  object
 9   capitalgain    15060 non-null  int64 
 10  capitalloss    15060 non-null  int64 
 11  hoursperweek   15060 non-null  int64 
 12  native         15060 non-null  object
 13  Salary         15060 non-null  object
dtypes: int64(5), object(9)
memory usage: 1.6+ MB
None
'''

##First five rows of the dataset
print(data2.head())
'''
Output:
age   workclass      education  ...  hoursperweek          native  Salary
0   25     Private           11th  ...            40   United-States   <=50K
1   38     Private        HS-grad  ...            50   United-States   <=50K
2   28   Local-gov     Assoc-acdm  ...            40   United-States    >50K
3   44     Private   Some-college  ...            40   United-States    >50K
4   34     Private           10th  ...            30   United-States   <=50K

[5 rows x 14 columns]
'''

## Checking Missing Values
print(data2.isnull().sum())
'''
Output:
age              0
workclass        0
education        0
educationno      0
maritalstatus    0
occupation       0
relationship     0
race             0
sex              0
capitalgain      0
capitalloss      0
hoursperweek     0
native           0
Salary           0
dtype: int64
'''
##There are no missing values

----------------------------------------------------------------------------------------------------

# 3) Separate Features and Target
X_train = data1.drop("Salary", axis=1)
y_train = data1["Salary"]

X_test = data2.drop("Salary", axis=1)
y_test = data2["Salary"]

##4) Preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

categorical_cols = ["workclass", "education", "maritalstatus", "occupation",
                    "relationship", "race", "sex", "native"]
numeric_cols = ["age", "educationno", "capitalgain", "capitalloss", "hoursperweek"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

# 5) Model Pipeline
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
pipeline = Pipeline([
    ("prep", preprocess),
    ("nb", GaussianNB())
])
--------------------------------------------------------------------------------------------------------------
# 6) Train the  Model
pipeline.fit(X_train, y_train)


# 7) Predictions and Evaluation on Test Data
y_pred_test = pipeline.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("\nTest Accuracy:", accuracy_score(y_test, y_pred_test))
##Output: Test Accuracy: 0.5849933598937583

print("\nConfusion Matrix (Test):\n", confusion_matrix(y_test, y_pred_test))
'''
Output:
Confusion Matrix (Test):
 [[5341 6019]
 [ 231 3469]]
 '''
 
print("\nClassification Report (Test):\n", classification_report(y_test, y_pred_test))
'''
Output:
classification_report(y_test, y_pred_test))

Classification Report (Test):
               precision    recall  f1-score   support

       <=50K       0.96      0.47      0.63     11360
        >50K       0.37      0.94      0.53      3700

    accuracy                           0.58     15060
   macro avg       0.66      0.70      0.58     15060
weighted avg       0.81      0.58      0.61     15060
'''
