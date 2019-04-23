# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:49:03 2019

@author: Michael
"""

import numpy as np
from pandas import read_excel
import keras
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler


df = read_excel("C:/Users/Michael/Documents/CS542/Final/carbon_nanotubes.xlsx")
X = df.iloc[:, 0:5]
X = X.values
y = df.iloc[:, 5:]
y = y.values

X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state = 42)


#use GridSearCV to find the best estimators

pipe_svr = Pipeline([('scl', StandardScaler()),
        ('reg', MultiOutputRegressor(SVR()))])

grid_param_svr = {'reg__estimator__gamma': [1e-3, 1e-4], 
                  'reg__estimator__C': [0.1, 1, 10], 
                  'reg__estimator__kernel': ['rbf', 'sigmoid']}

    
grid = GridSearchCV(pipe_svr, param_grid = grid_param_svr, 
                    cv = 10, scoring = 'neg_mean_squared_error')

grid = grid.fit(X_train, y_train)
print("The best parameters are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_))


kernel = ['rbf', 'linear', 'sigmoid']
gamma = [0.001, 0.0001]
C = [0.1, 1, 10]




model = SVR(kernel = 'sigmoid', C = 1, gamma = 0.001)
regr = MultiOutputRegressor(model)
regr.fit(X_train, y_train)
regr.score(X_train, y_train)

