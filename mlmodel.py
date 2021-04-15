# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 17:30:55 2021

@author: Anoushka
"""

import numpy as np
import pandas as pd
import math
import time
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sklearn.model_selection as RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import seaborn as sns
%matplotlib inline

df = pd.read_csv('C:/Users/Anoushka/Downloads/project2021.csv')

df = df.drop(['ConfirmedIndianNational','ConfirmedForeignNational','Time'], axis=1)
df.rename(columns={'State/UnionTerritory':'State'},inplace=True)


df = df.replace('Telengana','Telangana')
df = df.replace('Telengana***','Telangana')
df = df.replace('Telangana***','Telangana')
df = df.replace('Punjab***','Punjab')
df = df.replace('Chandigarh***','Chandigarh')
df = df.replace('Maharashtra***', 'Maharashtra')
df = df.replace('Nagaland#','Nagaland')
df = df.replace('Jharkhand#', 'Jharkhand')

df_row =df[(df['State'] == 'Cases being reassigned to states')].index 
df.drop(df_row,inplace=True)
df_row1 = df[(df['State'] == 'Unassigned')].index
df.drop(df_row1,inplace=True)
df_row2 = df[(df['State'] == 'Dadra and Nagar Haveli and Daman and Diu')].index
df.drop(df_row2,inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

label = le.fit_transform(df["State"])
data = df.drop("State",axis = 1)
data["States"] = label


X = data[['States','Cured', 'Confirmed']]
y = data[['Deaths']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape)
print(X_test.shape)

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression(normalize = True, fit_intercept=True)
linear_model.fit(X_train,y_train)
test_linear_predict = linear_model.predict(X_test)
linear_predict = linear_model.predict(X_train)

print('MAE: ',mean_absolute_error(test_linear_predict,y_test))
print('MSE: ',mean_squared_error(test_linear_predict,y_test))


plt.plot(y_test)
plt.plot(test_linear_predict)

r2_score = linear_model.score(X_test,y_test)
print(r2_score*100,'%')

import pickle 
import warnings

pickle.dump(linear_model,open('model.pkl','wb'))

















































