# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:12:47 2020

@author: Sayan Mondal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import cohen_kappa_score

data=pd.read_csv("C:/Users/Sayan Mondal/Desktop/heart-disease/heart.csv")

data.describe()

data.shape
data.columns
data.dtypes

data.isnull().sum()

# lets find the distribution of target variable...##
data["target"].value_counts()

## ploting the graph of Target...###
sns.set_style('darkgrid')
sns.countplot(x='target',data=data,palette="RdBu_r")

## correlation matrix....##
cormat=data.corr()

fig= plt.figure(figsize=(12,12))
sns.heatmap(cormat,annot=True,cmap="BuGn_r")
plt.show()

###plottting histogram##
data.hist()


## let's preprocess the data for machine understandable features...###
##...need to convert some variable to dummy variable....#
data=pd.get_dummies(data,columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])

### Scaling the data...###
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
col_scale=['age','trestbps','chol','thalach','oldpeak']
data[col_scale]=ss.fit_transform(data[col_scale])

y=data['target']
X=data.drop(['target'],axis=1)

## Logistic regression Algorithm....####

from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.20, random_state=42)

model=logreg.fit(X_train,y_train)

y_pred=model.predict(X_test)

n_errors=(y_pred!=y_test).sum()
print(n_errors)
## Number of errors=7..##

np.mean(y_test==y_pred)*100

print(classification_report(y_test,y_pred))

cohen_kappa_score(y_test,y_pred)
## .7703..##


#### Extreme Gradient Boosting Algo.....######

import xgboost as xgb 

XGBC=xgb.XGBClassifier() 

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.20, random_state=42)

model=XGBC.fit(X_train,y_train)

y_pred=model.predict(X_test)

n_errors=(y_pred!=y_test).sum()
print(n_errors)
## Number of errors=11..##
np.mean(y_test==y_pred)*100

print(classification_report(y_test,y_pred))

cohen_kappa_score(y_test,y_pred)
##.6402...##


## ExraTree Classifier algorithm...###
from sklearn.ensemble import ExtraTreesClassifier 
ET=ExtraTreesClassifier()

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.20, random_state=42)

model=ET.fit(X_train,y_train)

y_pred=model.predict(X_test)

n_errors=(y_pred!=y_test).sum()
print(n_errors)
## Number of errors=7..##
np.mean(y_test==y_pred)*100

print(classification_report(y_test,y_pred))

cohen_kappa_score(y_test,y_pred)
##.7695...##


######....Desision Tree Algorithm.....#####################
 from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.20, random_state=42)

model=clf.fit(X_train,y_train)

y_pred=model.predict(X_test)

n_errors=(y_pred!=y_test).sum()
print(n_errors)
## Number of errors=11..##
np.mean(y_test==y_pred)*100

print(classification_report(y_test,y_pred))

cohen_kappa_score(y_test,y_pred)
##.6402...##



###### CAT boost algorithm.....####

from catboost import CatBoostClassifier

cb=CatBoostClassifier()

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.30, random_state=42)

model=cb.fit(X_train,y_train)

y_pred=model.predict(X_test)

n_errors=(y_pred!=y_test).sum()
print(n_errors)
## Number of errors=16..##

np.mean(y_test==y_pred)*100

print(classification_report(y_test,y_pred))

cohen_kappa_score(y_test,y_pred)
## kappa score=0.6433....##


## ....So from the kappa Score and number of predicting errors, my finalised model is LogisticRegression...###










