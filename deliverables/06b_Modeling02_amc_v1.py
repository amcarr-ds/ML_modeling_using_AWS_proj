#!/usr/bin/env python
# coding: utf-8

# # ADS-508-01-SP23 Team 8: Final Project

# # Train model

# Much of the code is modified from `Fregly, C., & Barth, A. (2021). Data science on AWS: Implementing end-to-end, continuous AI and machine learning pipelines. O’Reilly.`

import boto3
from botocore.client import ClientError
import pandas as pd
import numpy as np
from pyathena import connect
from IPython.core.display import display, HTML
import missingno as msno
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso
import datetime as dt
import time
import sagemaker
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
import joblib
import os

# Citation: Hochberg, 2018; Shanmukh, 2021
m3v1_ls_pip = Pipeline([('si', SimpleImputer(strategy='median')),
                        ('ss', StandardScaler()),
                        ('ls', Lasso(random_state=1699))])

alpha_hparam = [.01, .05, .1, .5, 1, 2]
selection_hparam = ['cyclic', 'random']


m3v1_ls_grd = {'ls__alpha': alpha_hparam,
               'ls__selection': selection_hparam
           }

m3v1_ls = GridSearchCV(m3v1_ls_pip,
                       m3v1_ls_grd,
                       scoring='neg_root_mean_squared_error',
                       n_jobs=2,
                       refit=True,
                       verbose=2)

m3v1_ls.fit(train_x01, train_y01)

print(f'Best Estimator:\n{m3v1_ls.best_estimator_}')
print(f'Coefficients:\n{m3v1_ls.best_estimator_.named_steps["ls"].coef_}')

print(pd.DataFrame(m3v1_ls.cv_results_))

train_m3v1_ls_y01_pred = m3v1_ls.predict(train_x01)
print(train_m3v1_ls_y01_pred)

test_m3v1_ls_y01_pred = m3v1_ls.predict(test_x01)
print(test_m3v1_ls_y01_pred)

# Display evaluation metrics
# R-sq
train_m3v1_ls_r2 = r2_score(train_y01, train_m3v1_ls_y01_pred)
test_m3v1_ls_r2 = r2_score(test_y01, test_m3v1_ls_y01_pred)

print(f'Train R-sq:\n{train_m3v1_ls_r2}')
print(f'Test R-sq:\n{test_m3v1_ls_r2}')

# RMSE
train_m3v1_ls_rmse = mean_squared_error(train_y01, train_m3v1_ls_y01_pred, squared=False)
test_m3v1_ls_rmse = mean_squared_error(test_y01, test_m3v1_ls_y01_pred, squared=False)

print(f'Train RMSE:\n{train_m3v1_ls_rmse}')
print(f'Test RMSE:\n{test_m3v1_ls_rmse}')

# End timer script
end_time = dt.datetime.today()
time_elapse = end_time - start_time
print(f'End Time = {end_time}')
print(f'Script Time = {time_elapse}')


coef_intercept = np.hstack((m3v1_ls.best_estimator_.named_steps["ls"].coef_,
                            m3v1_ls.best_estimator_.named_steps["ls"].intercept_))
#print(coef_intercept)

coef_intercept_df01 = pd.DataFrame(coef_intercept)
#display(coef_intercept_df01)

train_x01_col_names = list(train_x01.columns)
train_x01_col_names.append('intercept')

train_x01_col_names_df01 = pd.DataFrame(train_x01_col_names)
#display(train_x01_col_names_df01)

model_params = pd.concat([train_x01_col_names_df01, coef_intercept_df01], axis=1)
display(model_params)
