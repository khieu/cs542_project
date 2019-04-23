# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:49:03 2019

@author: Michael
"""

import numpy as np
from pandas import read_excel
import keras
from sklearn import svm
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


df = read_excel("C:/Users/Michael/Documents/CS542/Final/carbon_nanotubes.xlsx")
X = df.iloc[:, 0:5]
X = X.values
y = df.iloc[:, 5:]
y = y.values

X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state = 42)


# Try multiple Kernel on SVR
model = svm.SVR(kernel = 'rbf')
regr = MultiOutputRegressor(model)
regr.fit(X_train, y_train)
regr.score(X_train, y_train)
y_pred = regr.predict(X_test)

