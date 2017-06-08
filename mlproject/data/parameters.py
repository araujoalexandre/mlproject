
# classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier

# Regression
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor 
from sklearn.linear_model import RANSACRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor

from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Hinge
from sklearn.linear_model import Huber
from sklearn.linear_model import Lasso
from sklearn.linear_model import Lars
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.kernel_ridge import KernelRidge


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
    #     'params': {
    #     }
    # }
    # models += [Sklearn(params, paths)]


    # params = {
    #     'model': Ridge(),
    #     'ext': ext,
    #     'params': {
    #         'max_iter': max_iter,
    #     }
    # }
    # models += [Sklearn(params, paths)]

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

    return models





