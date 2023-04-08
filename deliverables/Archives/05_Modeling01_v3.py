import sys
import subprocess
import pandas as pd
import numpy as np
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.externals import joblib
import sagemaker
import datetime as dt
import time
import os
from io import BytesIO

if __name__ == "__main__":
    # Start timer script
    start_time = dt.datetime.today()

    # Parse input arguments
    train_x01_uri = sys.argv[1]
    train_y01_uri = sys.argv[2]

    # Load input data from S3
    train_x01 = pd.read_csv(train_x01_uri)
    train_y01 = np.load(train_y01_uri)

    # Citation: Hochberg, 2018; Shanmukh, 2021
    m3v2_ls_pip = Pipeline([('si', SimpleImputer(strategy='median')),
                            ('ss', StandardScaler()),
                            ('ls', Lasso(random_state=1699))])

    alpha_hparam = Real(1e-3, 1e3, prior='log-uniform')
    selection_hparam = Categorical(['cyclic', 'random'])
    max_iter_hparam = Integer(100, 5000, prior='log-uniform')
    warm_start_hparam = Categorical([False, True])

    m3v2_ls_grd = {'ls__alpha': alpha_hparam,
                   'ls__selection': selection_hparam,
                   'ls__max_iter': max_iter_hparam,
                   'ls__warm_start': warm_start_hparam
               }

    m3v2_ls = BayesSearchCV(m3v2_ls_pip,
                           m3v2_ls_grd,
                           scoring='neg_root_mean_squared_error',
                            cv=5,
                           n_jobs=2,
                           refit=True,
                           verbose=2)

    m3v2_ls.fit(train_x01, train_y01)
