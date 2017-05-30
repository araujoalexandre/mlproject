"""
    fscipt to generate the dataset
"""
from os.path import join

from kaggle.helper.GenerateWrapper import GenerateWrapper

from kaggle.utils import ParametersSpace
from kaggle.utils import pickle_dump, pickle_load
from kaggle.utils import pprint

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold


################################
#####    Globals params    #####
################################

params = dict(
    #project
    project_path  = 'test',
    data_path     = join('test', 'data'),
    seed          = 123456,
    nan_value     = -1,

    # ids
    id_train      = 'id',
    id_test       = 'id_test',
    
    # target name
    target_train  = 'target',
    target_test   = 'test_target',

    # instance weights
    weights       = '',

    # groups
    group_train   = '',
    group_test    = '',

    # validation
    n_folds       = 5,
    shuffle       = True,
)

def dataset(splits):
    """
        Create your dataset here !
    """
    train_path = join(params['data_path'], 'train', 'train.csv')
    test_path = join(params['data_path'], 'test', 'test.csv')
    
    df_train = pd.read_csv(train_path, nows=20000)
    df_test = pd.read_csv(test_path, nows=20000)

    return df_train, df_test


def validation_splits(fold, y, suffle=True, seed=1234, groups=None):
    """
        define the vaidation strategy in this function
    """

    # gkf = GroupKFold(n_splits=fold)
    # split = list(gkf.split(X, y, groups=groups))

    skf = StratifiedKFold(n_splits=fold, shuffle=suffle, random_state=seed)
    split = list(skf.split(y, y))

    # kf = KFold(n_splits=fold, shuffle=suffle, random_state=seed)
    # split = list(kf.split(X, y))

    return split