"""
__file__

    parameters.py

__description__

    This file provides global parameter configurations for the project.

__author__

    Araujo Alexandre < aaraujo001@gmail.com >

"""

import os, copy

from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.naive_bayes import *

from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor

from sklearn.metrics import mean_absolute_error

from kaggle.wrapper.xgboost import XGBoost
from kaggle.wrapper.sklearn import Sklearn
from kaggle.wrapper.lightgbm import LightGBM
from kaggle.wrapper.liblinear import Liblinear
from kaggle.wrapper.libffm import LibFFM

import numpy as np

from kaggle.utils.functions import target_transform, pickle_load


#################################################
#####    Parameters space for train file    #####
#################################################

def get_params():

    paths = {
        'folder_path': os.getcwd()
    }

    models = []

    

    # max_iter = 1000
    # ext = 'npz'

    # params = {
    #     'model': LogisticRegression(),
    #     'ext': ext,
    #     'n_jobs': -1,
    #     'params': {
    #         'solver': 'sag',
    #         'max_iter': max_iter,
    #     }
    # }
    # models.append(Sklearn(params, paths))



    #############################################
    #####    Parameter Space for XGBoost    #####
    #############################################

    ext = 'xgb'
    predict_option = 'best_ntree_limit'
    booster = {
        'num_boost_round': 600, 
        'early_stopping_rounds': 50, 
        'verbose_eval': 30,
    }

    params = {
        'ext': ext,
        'predict_option': predict_option,
        'booster': booster,
        'params': {
            'booster': 'gbtree',
            'nthread': 12,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': 0.05,
            'max_depth': 5,
            'subsample': 0.5,
            'colsample_bytree': 0.7,
            'min_child_weight': 1,
            # 'scale_pos_weight': 0.445,
            'scale_pos_weight': 0.339,
            # 'scale_pos_weight': 0.1976,
            'silent': 1,
        },
    }
    models.append(XGBoost(params, paths))

    # params = {
    #     'ext': ext,
    #     'predict_option': predict_option,
    #     'booster': booster,
    #     'params': {
    #         'booster': 'gbtree',
    #         'nthread': 11,
    #         'objective': 'multi:softprob',
    #         'eval_metric': 'mlogloss',
    #         'eta': 0.05,
    #         'max_depth': 20,
    #         'subsample': 0.9,
    #         'colsample_bytree': 0.3,
    #         'min_child_weight': 1,
    #         'num_class': 3,
    #         'silent': 1,
    #     },
    # }
    # models.append(XGBoost(params, paths))


    # params['booster']['obj'] = logregobj
    # models.append(XGBoost(params, paths))


    ###############################################
    #####    Parameters space for LightGBM    #####
    ###############################################

    
    # ext = 'libsvm'

    # params = {
    #     'ext': ext,
    #     'params': {
    #         'application': 'regression',
    #         'boosting': 'gbdt', 
    #         'learning_rate': 0.01, 
    #         'task': 'train',
    #         'num_iterations': 20000,
    #         'early_stopping_round': 30,
    #         'metric': 'l2',

    #         'max_depth': 13,
    #         'feature_fraction': 1,
    #         # 'min_data_in_leaf': 320,
    #         # 'min_gain_to_split': 0,
    #         'lambda_l1': 1.7,
    #         # 'lambda_l2': 1.6,
    #         # 'bagging_freq': 1, 
    #         # 'num_leaves': 105, 
    #         # 'bagging_fraction': 1, 
    #         # 'min_sum_hessian_in_leaf': 10
    #     },
    # }
    # models.append(LightGBM(params, paths))


    ##################################################################
    #####    Parameter Space for Scikit Learn Ensemble Models    #####
    ##################################################################

    n_estimators = 1000
    n_jobs = 12
    ext = 'npz'

    # params = {
    #     'model': RandomForestClassifier(),
    #     'n_jobs': n_jobs,
    #     'ext': ext,
    #     'params': {
    #         'n_estimators': n_estimators,
    #         'criterion': 'entropy',
    #     },
    # }
    # models.append(Sklearn(params, paths))

    # params = {
    #     'model': ExtraTreesClassifier(),
    #     'n_jobs': n_jobs,
    #     'ext': ext,
    #     'params': {
    #         'n_estimators': n_estimators,
    #         'criterion': 'entropy',
    #     },
    # }
    # models.append(Sklearn(params, paths))


    # params = {
    #     'model': RandomForestClassifier(),
    #     'n_jobs': n_jobs,
    #     'ext': ext,
    #     'params': {
    #         'n_estimators': n_estimators,
    #         'criterion': 'gini',
    #     },
    # }
    # models.append(Sklearn(params, paths))

    # params = {
    #     'model': ExtraTreesClassifier(),
    #     'n_jobs': n_jobs,
    #     'ext': ext,
    #     'params': {
    #         'n_estimators': n_estimators,
    #         'criterion': 'gini',
    #     },
    # }
    # models.append(Sklearn(params, paths))



    # ##########################################################
    # #####    Parameter Space for Scikit Linear Models    #####
    # ##########################################################

    max_iter = 1000
    ext = 'npz'

    # params = {
    #     'model': LogisticRegression(),
    #     'ext': ext,
    #     'n_jobs': -1,
    #     'params': {
    #     }
    # }
    # models.append(Sklearn(params, paths))


    # params = {
    #     'model': Ridge(),
    #     'ext': ext,
    #     'params': {
    #         'max_iter': max_iter,
    #     }
    # }
    # models.append(Sklearn(params, paths))

    # params = {
    #     'model': KernelRidge(),
    #     'ext': ext,
    #     'params': {
    #         'max_iter': max_iter,
    #     }
    # }
    # models.append(Sklearn(params, paths))

    # params = {
    #     'model': HuberRegressor(),
    #     'ext': ext,
    #     'params': {
    #         'max_iter': max_iter,
    #     }
    # }
    # models.append(Sklearn(params, paths))

    # params = {
    #     'model': ARDRegression(),
    #     'ext': ext,
    #     'params': {
    #         'max_iter': max_iter,
    #     }
    # }
    # models.append(Sklearn(params, paths))

    # params = {
    #     'model': BayesianRidge(),
    #     'ext': ext,
    #     'params': {
    #         'max_iter': max_iter,
    #     }
    # }
    # models.append(Sklearn(params, paths))

    # params = {
    #     'model': ElasticNet(),
    #     'ext': ext,
    #     'params': {
    #         'max_iter': max_iter,
    #     }
    # }
    # models.append(Sklearn(params, paths))

    # params = {
    #     'model': SGDRegressor(),
    #     'ext': ext,
    #     'params': {
    #         'max_iter': max_iter,
    #     }
    # }
    # models.append(Sklearn(params, paths))

    # params = {
    #     'model': LinearRegression(),
    #     'ext': ext,
    #     'n_jobs': -1,
    #     'params': {
    #     }
    # }
    # models.append(Sklearn(params, paths))




    # params = {
    #     'model': Lasso(),
    #     'ext': ext,
    #     'params': {
    #         'max_iter': max_iter, 
    #     }
    # }
    # models.append(Sklearn(params, paths))


    #################################################
    #####    Parameter Space for Naive Bayes    #####
    #################################################

    # ext = 'npz'

    # params = {
    #     'model': BernoulliNB(),
    #     'ext': ext,
    #     'params': {
    #     }
    # }
    # models.append(Sklearn(params, paths))


    # params = {
    #     'model': GaussianNB(),
    #     'ext': ext,
    #     'params': {
    #     }
    # }
    # models.append(Sklearn(params, paths))


    # params = {
    #     'model': MultinomialNB(),
    #     'ext': ext,
    #     'params': {
    #     }
    # }
    # models.append(Sklearn(params, paths))


 
    ################################################
    #####    Parameters space for LIBLINEAR    #####
    ################################################

    # ext = 'npz'

    # params = {
    #     'ext': ext,
    #     'params': {
    #         'type_solver': 11,
    #         # 'cost': 1,
    #         # 'epsilon_p': 0.1,
    #         # 'epsilon_e': 0.01,
    #         # 'bias': -1,
    #         'silent': 0,
    #         # 'predict_opt': 1
    #     },
    # }
    # # models.append(Liblinear(params, paths))


    ###################################
    #####    Sklearn Neighbors    #####
    ###################################
    
    # ext = 'npz'
    # n_jobs = -1

    # params = {
    #     'model': KNeighborsRegressor(),
    #     'n_jobs': n_jobs,
    #     'ext': ext,
    #     'params': {
    #     },
    # }
    # # models.append(Sklearn(params, paths))

    # params = {
    #     'model': RadiusNeighborsRegressor(),
    #     'n_jobs': n_jobs,
    #     'ext': ext,
    #     'params': {
    #     },
    # }
    # # models.append(Sklearn(params, paths))
    


    #############################################
    #####    Parameters space for LIBFFM    #####
    #############################################
    
    # ext = 'libffm'
    # ext = 'custom'

    # params = {
    #     'application':'classification',
    #     'ext': ext,
    #     'params': {
    #         'lambda': 0.00002,
    #         'factor': 4,
    #         'iteration': 20,
    #         'eta': 0.3,
    #     }
    # }
    # models.append(LibFFM(params, paths))

    #####################################################
    #####    Parameters space for neural network    #####
    #####################################################

    # params = {
    #     'type':'neural_network',
    #     'params': {
    #     }
    # }

    return models





