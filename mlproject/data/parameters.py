
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.linear_model import *
from sklearn.naive_bayes import *

# Wrapper
from mlproject.wrapper import LibFFMWrapper
from mlproject.wrapper import LiblinearWrapper
from mlproject.wrapper import LightGBMWrapper
from mlproject.wrapper import SklearnWrapper
from mlproject.wrapper import XGBoostWrapper


#################################################
#####    Parameters space for train file    #####
#################################################

def get_models_wrapper():

    models = []

    #############################################
    #####    Parameter Space for XGBoost    #####
    #############################################
    """
        params dict takes a ext, XXX 
        URL to params
    """

    ext = 'xgb'
    nthread = 12
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
            'nthread': nthread,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': 0.05,
            'silent': 1,
        },
    }
    models += [XGBoostWrapper(params)]

    ###############################################
    #####    Parameters space for LightGBM    #####
    ###############################################
    """
        XXX
    """
    
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
    #         'lambda_l1': 1.7,
    #     },
    # }
    # models += [LightgbmWrapper(params)]


    ##################################################################
    #####    Parameter Space for Scikit-Learn Ensemble Models    #####
    ##################################################################
    """
        XXX
    """

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
    # models += [SklearnWrapper(params)]

    # params = {
    #     'model': ExtraTreesClassifier(),
    #     'n_jobs': n_jobs,
    #     'ext': ext,
    #     'params': {
    #         'n_estimators': n_estimators,
    #         'criterion': 'entropy',
    #     },
    # }
    # models += [SklearnWrapper(params)]

    # params = {
    #     'model': RandomForestClassifier(),
    #     'n_jobs': n_jobs,
    #     'ext': ext,
    #     'params': {
    #         'n_estimators': n_estimators,
    #         'criterion': 'gini',
    #     },
    # }
    # models += [SklearnWrapper(params)]

    # params = {
    #     'model': ExtraTreesClassifier(),
    #     'n_jobs': n_jobs,
    #     'ext': ext,
    #     'params': {
    #         'n_estimators': n_estimators,
    #         'criterion': 'gini',
    #     },
    # }
    # models += [SklearnWrapper(params)]


    ##########################################################
    #####    Parameter Space for Scikit Linear Models    #####
    ##########################################################

    max_iter = 1000
    ext = 'npz'

    # params = {
    #     'model': LogisticRegression(),
    #     'ext': ext,
    #     'params': {
    #     }
    # }
    # models += [SklearnWrapper(params)]

    # params = {
    #     'model': Ridge(),
    #     'ext': ext,
    #     'params': {
    #         'max_iter': max_iter,
    #     }
    # }
    # models += [SklearnWrapper(params)]

    # params = {
    #     'model': KernelRidge(),
    #     'ext': ext,
    #     'params': {
    #         'max_iter': max_iter,
    #     }
    # }
    # models += [SklearnWrapper(params)]

    # params = {
    #     'model': HuberRegressor(),
    #     'ext': ext,
    #     'params': {
    #         'max_iter': max_iter,
    #     }
    # }
    # models += [SklearnWrapper(params)]

    # params = {
    #     'model': ARDRegression(),
    #     'ext': ext,
    #     'params': {
    #         'max_iter': max_iter,
    #     }
    # }
    # models += [SklearnWrapper(params)]

    # params = {
    #     'model': BayesianRidge(),
    #     'ext': ext,
    #     'params': {
    #         'max_iter': max_iter,
    #     }
    # }
    # models += [SklearnWrapper(params)]

    # params = {
    #     'model': ElasticNet(),
    #     'ext': ext,
    #     'params': {
    #         'max_iter': max_iter,
    #     }
    # }
    # models += [SklearnWrapper(params)]

    # params = {
    #     'model': SGDRegressor(),
    #     'ext': ext,
    #     'params': {
    #         'max_iter': max_iter,
    #     }
    # }
    # models += [SklearnWrapper(params)]

    # params = {
    #     'model': LinearRegression(),
    #     'ext': ext,
    #     'n_jobs': -1,
    #     'params': {
    #     }
    # }
    # models += [SklearnWrapper(params)]

    # params = {
    #     'model': Lasso(),
    #     'ext': ext,
    #     'params': {
    #         'max_iter': max_iter, 
    #     }
    # }
    # models += [SklearnWrapper(params)]


    #################################################
    #####    Parameter Space for Naive Bayes    #####
    #################################################

    ext = 'npz'

    # params = {
    #     'model': BernoulliNB(),
    #     'ext': ext,
    #     'params': {
    #     }
    # }
    # models += [SklearnWrapper(params)]


    # params = {
    #     'model': GaussianNB(),
    #     'ext': ext,
    #     'params': {
    #     }
    # }
    # models += [SklearnWrapper(params)]


    # params = {
    #     'model': MultinomialNB(),
    #     'ext': ext,
    #     'params': {
    #     }
    # }
    # models += [SklearnWrapper(params)]


    ###################################
    #####    Sklearn Neighbors    #####
    ###################################
    
    ext = 'npz'
    n_jobs = -1

    # params = {
    #     'model': KNeighborsRegressor(),
    #     'n_jobs': n_jobs,
    #     'ext': ext,
    #     'params': {
    #     },
    # }
    # models += [SklearnWrapper(params)]

    # params = {
    #     'model': RadiusNeighborsRegressor(),
    #     'n_jobs': n_jobs,
    #     'ext': ext,
    #     'params': {
    #     },
    # }
    # models += [SklearnWrapper(params)]


    ################################################
    #####    Parameters space for LIBLINEAR    #####
    ################################################

    # ext = 'npz'

    # params = {
    #     'ext': ext,
    #     'params': {
    #         'type_solver': 11,
    #         'silent': 0,
    #     },
    # }
    # models += [LiblinearWrapper(params)]


    #############################################
    #####    Parameters space for LIBFFM    #####
    #############################################
    
    # ext = 'ffm'

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
    # models += [LibFFMWrapper(params)]

    return models





