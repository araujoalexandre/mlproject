"""
    fscipt to generate the dataset
"""
from os import getcwd
from glob import glob
from os.path import join, dirname, basename
from itertools import product, combinations

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold

# regression task
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error

# classification task
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import average_precision_score

from mlproject.utils.parameters import ParametersSpace
from mlproject.utils.project import ProjectPath, project_path
from mlproject.utils.serialization import pickle_load

load = [ 'define_params', 'create_dataset', 'validation_splits', 'metric',
    'make_submit', 'target_preprocess', 'target_postprocess' ]

################################################################################
#####                      Parameters for the project                      #####
################################################################################
"""
    XXX other comment
"""

def define_params():
    params = ParametersSpace(

        # path to the project and to the data folder
        project_path = project_path(getcwd()),
        data_path = join(project_path(getcwd()), 'data'),

        # names of dataset
        # the function will generate train_path and test_path attributes
        train_name = 'train.csv',
        test_name = 'test.csv',

        # name of the id features in train and test set
        id_train = 'id',
        id_test = 'id',

        # name of the target feature in the dataset or path to a pickle file
        # or a numpy array 
        # explain about validation
        target_train = 'target',
        target_test = 'target',

        # set weights of instances
        weights_train = None,
        weights_test = None,

        # set group size / query size (used for ranking)
        groups_train = None,
        groups_test = None,

        # number of folds for validation strategy
        n_folds = 5,

        # seed value for the project
        seeds = 123456,

        # value to fill for nan value in dataset
        missing = -1,
    )
    return params


################################################################################
#####                    Function to create the dataset                    #####
################################################################################
"""
    XXX
"""

def target_encoding(df_train, df_test, splits, name_feature, cols, group_limit=5, target='target'):
    feature = 'target_encoding__{}'.format(name_feature)
    for tr_ix, va_ix in splits:
        # make mapper
        mapper = df_train.loc[tr_ix].groupby(name_feature)[target].agg(['mean', 'count'])
        mapper = mapper[mapper['count'] > group_limit]['mean'].copy()        
        # map proba to feature value
        df_train.loc[va_ix, feature] = df_train.loc[va_ix][name_feature].map(mapper)
    # make mapper with all training set
    mapper = df_train.groupby(name_feature)[target].agg(['mean', 'count'])
    mapper = mapper[mapper['count'] > group_limit]['mean'].copy()   
    # map proba to feature value
    df_test[feature] = df_test[name_feature].map(mapper)
    # fillna with mean
    fill_value = df_train[feature].mean()
    df_train[feature].fillna(fill_value, inplace=True)
    df_test[feature].fillna(fill_value, inplace=True)
    return df_train, df_test

def create_dataset(splits):
    """
        Create your dataset here !
    """
    project = ProjectPath(project_path(getcwd()))
    params = define_params()

    train_path = join(project.data.train.raw(), params.train_name)
    test_path = join(project.data.test.raw(), params.test_name)
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # do feature engineering here !

    return df_train, df_test

################################################################################
#####                    Define your validation strategy                   #####
################################################################################
"""
    XXX
"""

def validation_splits(nfold, y, seed, groups=None):
    """
        Define the vaidation strategy here !
    """

    # gkf = GroupKFold(n_splits=nfold)
    # split = list(gkf.split(y, y, groups=groups))

    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
    split = list(skf.split(y, y))

    # kf = KFold(n_splits=nfold, shuffle=True, random_state=seed)
    # split = list(kf.split(y, y))

    return split

################################################################################
#####              Define your metric for evalution strategy               #####
################################################################################
"""
    XXX
"""

def metric(y, yhat, weights=None, groups=None):
    """
        create your metric here !
    """
    return log_loss(y, yhat)

################################################################################
#####        Function to create the submission file for competitions       #####
################################################################################
"""
    XXX
"""

def make_submit(path, id_test, yhat, model_name, date, score_va, score_test):
    """
        Create the submission file here !
    """
    df_submit = pd.DataFrame({'id': id_test, 'target': yhat})
    name = ("submit_{name}_{date:%m%d_%H%M}_"
            "{score_va:.5f}_{score_test:.5f}.csv.gz").format(name=model_name, 
                            date=date, score_va=score_va, score_test=score_test)
    out = join(path, name)
    df_submit.to_csv(out, index=False, compression='gzip')

################################################################################
#####                  Function to pre-process the target                  #####
################################################################################
"""
    XXX
"""

def target_preprocess(y):
    """
        Create target pre-processing function here !
    """
    # trainsform y

    return y


################################################################################
#####                  Function to post-process the target                 #####
################################################################################
"""
    XXX
"""

def target_postprocess(y):
    """
        Create target post-processing function here !
    """
    # transform y

    return y