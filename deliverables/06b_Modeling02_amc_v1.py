#!/usr/bin/env python
# coding: utf-8

# # ADS-508-01-SP23 Team 8: Final Project

# # Train model

# Much of the code is modified from `Fregly, C., & Barth, A. (2021). Data science on AWS: Implementing end-to-end, continuous AI and machine learning pipelines. O’Reilly.`

# ## Install missing dependencies
# 
# [PyAthena](https://pypi.org/project/PyAthena/) is a Python DB API 2.0 (PEP 249) compliant client for Amazon Athena.

# In[2]:


get_ipython().system('pip install --disable-pip-version-check -q PyAthena==2.1.0')
get_ipython().system('pip install --disable-pip-version-check -q sagemaker-experiments==0.1.26')
get_ipython().system('pip install missingno')


# ## Globally import libraries

# In[3]:


import boto3
from botocore.client import ClientError
import pandas as pd
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


# ## Instantiate AWS SageMaker and S3 sessions

# In[4]:


session = boto3.session.Session()
role = sagemaker.get_execution_role()
region = session.region_name
sagemaker_session = sagemaker.Session()
def_bucket = sagemaker_session.default_bucket()
bucket = 'sagemaker-us-east-ads508-sp23-t8'

s3 = boto3.Session().client(service_name="s3",
                            region_name=region)

sm = boto3.Session().client(service_name="sagemaker",
                            region_name=region)


# In[5]:


setup_s3_bucket_passed = False
ingest_create_athena_db_passed = False
ingest_create_athena_table_tsv_passed = False


# In[6]:


print(f"Default bucket: {def_bucket}")
print(f"Public T8 bucket: {bucket}")


# ## Verify S3 Bucket Creation

# In[7]:


get_ipython().run_cell_magic('bash', '', '\naws s3 ls s3://${bucket}/')


# In[8]:


response = None

try:
    response = s3.head_bucket(Bucket=bucket)
    print(response)
    setup_s3_bucket_passed = True
except ClientError as e:
    print(f"[ERROR] Cannot find bucket {bucket} in {response} due to {e}.")


# In[9]:


get_ipython().run_line_magic('store', 'setup_s3_bucket_passed')


# ## Pass in ABT from CSV

# In[10]:


s3_abt_csv_path = f"s3://{def_bucket}/team_8_data/abt/abt_encoded_df01.csv"
abt_encoded_df01 = pd.read_csv(s3_abt_csv_path)


# In[11]:


y01 = ['childpoverty']
abt_encoded_y01_vc01 = abt_encoded_df01[y01].to_numpy()
print(abt_encoded_y01_vc01.shape)
display(abt_encoded_y01_vc01[0:11])
abt_encoded_x01_df01 = abt_encoded_df01.drop(y01, axis=1)
print(abt_encoded_x01_df01.shape)
display(abt_encoded_x01_df01.head(11))

abt_encoded_x01_df01['boroughs'] = abt_encoded_x01_df01['borough_bronx'].astype(int).astype(str) + abt_encoded_x01_df01['borough_brooklyn'].astype(int).astype(str) + abt_encoded_x01_df01['borough_manhattan'].astype(int).astype(str) + abt_encoded_x01_df01['borough_queens'].astype(int).astype(str) + abt_encoded_x01_df01['borough_staten island'].astype(int).astype(str)
display(abt_encoded_x01_df01.head(5))
train_x01, test_x01, train_y01, test_y01 = train_test_split(abt_encoded_x01_df01,
                                                            abt_encoded_y01_vc01,
                                                            test_size=.2,
                                                            stratify=abt_encoded_x01_df01[['boroughs']],
                                                            shuffle=True,
                                                            random_state=1699)

train_x01 = train_x01.drop(['boroughs', 'poverty'], axis=1)
test_x01 = test_x01.drop(['boroughs', 'poverty'], axis=1)

train_y01 = train_y01.ravel()
test_y01 = test_y01.ravel()


print(f'{train_x01.shape}')
print(f'{train_y01.shape}')
print(f'\n{test_x01.shape}')
print(f'{test_y01.shape}')


# ## Model Training using Grid search with 5-fold cross-validation

# ### Random Forests

# # Start timer script
# start_time = dt.datetime.today()
# 
# # Citation: Hochberg, 2018; Shanmukh, 2021
# m1v1_pip = Pipeline([('si', SimpleImputer(strategy='median')),
#                      ('ss', StandardScaler()),
#                      ('rf', RandomForestRegressor(n_jobs=2, random_state=1699))])
# 
# n_est_hparam = [1, 10, 100, 200]
# max_depth_hparam = [5, 10, None]
# min_samp_leaf_hparam = [1, 75, 105]
# max_feat_hparam = ['sqrt', 'log2']
# 
# #n_est_hparam = [1, 10]
# #max_depth_hparam = [5, 10]
# #min_samp_leaf_hparam = [1, 75]
# #max_feat_hparam = ['sqrt']
# 
# m1v1_grd = {'rf__n_estimators': n_est_hparam,
#             'rf__max_depth': max_depth_hparam,
#             'rf__min_samples_leaf': min_samp_leaf_hparam,
#             'rf__max_features': max_feat_hparam
#            }
# 
# m1v1_rf = GridSearchCV(m1v1_pip,
#                        m1v1_grd,
#                        scoring='neg_root_mean_squared_error',
#                        n_jobs=2,
#                        refit=True,
#                        verbose=2)
# 
# m1v1_rf.fit(train_x01, train_y01)
# 
# print(f'Best Estimator:\n{m1v1_rf.best_estimator_}')
# 
# print(pd.DataFrame(m1v1_rf.cv_results_))
# 
# train_m1v1_y01_pred = m1v1_rf.predict(train_x01)
# print(train_m1v1_y01_pred)
# 
# test_m1v1_y01_pred = m1v1_rf.predict(test_x01)
# print(test_m1v1_y01_pred)
# 
# # Display evaluation metrics
# # R-sq
# train_m1v1_rf_r2 = r2_score(train_y01, train_m1v1_y01_pred)
# test_m1v1_rf_r2 = r2_score(test_y01, test_m1v1_y01_pred)
# 
# print(f'Train R-sq:\n{train_m1v1_rf_r2}')
# print(f'Test R-sq:\n{test_m1v1_rf_r2}')
# 
# # RMSE
# train_m1v1_rf_rmse = mean_squared_error(train_y01, train_m1v1_y01_pred, squared=False)
# test_m1v1_rf_rmse = mean_squared_error(test_y01, test_m1v1_y01_pred, squared=False)
# 
# print(f'Train RMSE:\n{train_m1v1_rf_rmse}')
# print(f'Test RMSE:\n{test_m1v1_rf_rmse}')
# 
# # End timer script
# end_time = dt.datetime.today()
# time_elapse = end_time - start_time
# print(f'End Time = {end_time}')
# print(f'Script Time = {time_elapse}')

# s3_m1v1_rf_pqt_base_path = f"../models"
# 
# if not os.path.exists(s3_m1v1_rf_pqt_base_path):
#     os.makedirs(s3_m1v1_rf_pqt_base_path)
# 
# s3_m1v1_rf_pqt_path = os.path.join(s3_m1v1_rf_pqt_base_path,
#                                    'm1v1_rf.parquet')
# 
# # save the model to disk using joblib
# joblib.dump(m1v1_rf,
#             s3_m1v1_rf_pqt_path)
# 
# # load the saved model from disk using joblib
# m1v1_rf_fitted = joblib.load(s3_m1v1_rf_pqt_path)

# ### Neural Network

# # Start timer script
# start_time = dt.datetime.today()
# 
# # Citation: Hochberg, 2018; Shanmukh, 2021
# m2v1_pip = Pipeline([('si', SimpleImputer(strategy='median')),
#                      ('ss', StandardScaler()),
#                      ('nn', MLPRegressor(random_state=1699))])
# 
# nodes_h = 3
# predictors_p = 49
# 
# hidden_layer_sizes_hparam = [[100,],
#                              [(nodes_h*(predictors_p+1))+nodes_h+1,],
#                              [50, 50]
#                             ]
# activation_hparam = ['logistic', 'relu']
# solver_hparam = ['adam']
# alpha_hparam = [.0001, .0005, .001]
# learn_rate_hparam = ['constant', 'invscaling']
# 
# #hidden_layer_sizes_hparam = [[100,]]
# #activation_hparam = ['relu']
# #solver_hparam = ['adam']
# #alpha_hparam = [.0001]
# #learn_rate_hparam = ['invscaling']
# 
# m2v1_grd = {'nn__hidden_layer_sizes': hidden_layer_sizes_hparam,
#             'nn__activation': activation_hparam,
#             'nn__solver': solver_hparam,
#             'nn__alpha': alpha_hparam,
#             'nn__learning_rate': learn_rate_hparam
#            }
# 
# m2v1_nn = GridSearchCV(m2v1_pip,
#                        m2v1_grd,
#                        scoring='neg_root_mean_squared_error',
#                        n_jobs=2,
#                        refit=True,
#                        verbose=2)
# 
# m2v1_nn.fit(train_x01, train_y01)
# 
# print(f'Best Estimator:\n{m2v1_nn.best_estimator_}')
# 
# print(pd.DataFrame(m2v1_nn.cv_results_))
# 
# train_m2v1_y01_pred = m2v1_nn.predict(train_x01)
# print(train_m2v1_y01_pred)
# 
# test_m2v1_y01_pred = m2v1_nn.predict(test_x01)
# print(test_m2v1_y01_pred)
# 
# # Display evaluation metrics
# # R-sq
# train_m2v1_nn_r2 = r2_score(train_y01, train_m2v1_y01_pred)
# test_m2v1_nn_r2 = r2_score(test_y01, test_m2v1_y01_pred)
# 
# print(f'Train R-sq:\n{train_m2v1_nn_r2}')
# print(f'Test R-sq:\n{test_m2v1_nn_r2}')
# 
# # RMSE
# train_m2v1_nn_rmse = mean_squared_error(train_y01, train_m2v1_y01_pred, squared=False)
# test_m2v1_nn_rmse = mean_squared_error(test_y01, test_m2v1_y01_pred, squared=False)
# 
# print(f'Train RMSE:\n{train_m2v1_nn_rmse}')
# print(f'Test RMSE:\n{test_m2v1_nn_rmse}')
# 
# # End timer script
# end_time = dt.datetime.today()
# time_elapse = end_time - start_time
# print(f'End Time = {end_time}')
# print(f'Script Time = {time_elapse}')

# s3_m2v1_nn_pqt_base_path = f"../models"
# 
# if not os.path.exists(s3_m2v1_nn_pqt_base_path):
#     os.makedirs(s3_m2v1_nn_pqt_base_path)
# 
# s3_m2v1_nn_pqt_path = os.path.join(s3_m2v1_nn_pqt_base_path,
#                                    'm2v1_nn.parquet')
# 
# # save the model to disk using joblib
# joblib.dump(m2v1_nn,
#             s3_m2v1_nn_pqt_path)
# 
# # load the saved model from disk using joblib
# m2v1_nn_fitted = joblib.load(s3_m2v1_nn_pqt_path)

# ### Lasso

# In[12]:


# Start timer script
start_time = dt.datetime.today()

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


# In[13]:


import numpy as np
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


# ## Release Resources

# In[14]:


get_ipython().run_cell_magic('html', '', '\n<p><b>Shutting down your kernel for this notebook to release resources.</b></p>\n<button class="sm-command-button" data-commandlinker-command="kernelmenu:shutdown" style="display:none;">Shutdown Kernel</button>\n        \n<script>\ntry {\n    els = document.getElementsByClassName("sm-command-button");\n    els[0].click();\n}\ncatch(err) {\n    // NoOp\n}    \n</script>')


# In[15]:


get_ipython().run_cell_magic('javascript', '', '\ntry {\n    Jupyter.notebook.save_checkpoint();\n    Jupyter.notebook.session.delete();\n}\ncatch(err) {\n    // NoOp\n}')

