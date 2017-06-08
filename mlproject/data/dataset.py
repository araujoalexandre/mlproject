"""
    fscipt to generate the dataset
"""
from os import getcwd
from os.path import join
from mlproject.utils import ParametersSpace
from mlproject.utils import project_path

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import KFold
    from sklearn.model_selection import GroupKFold
    SKLEARN_INSTALLED = True
except ImportError:
    SKLEARN_INSTALLED = False

################################
#####    Globals params    #####
################################

params = ParametersSpace(

    # path to the project and to the data folder
    project_path = project_path(getcwd()),
    data_path = join(project_path(getcwd()), 'data'),
    
    # names of dataset
    # the function will generate train_path and test_path attributes
    train_name = 'train.csv',
    test_name = 'test.csv',

    # seed value for the project
    seed = 123456,

    # value to fill for nan value in dataset
    missing = -1,

    # name of the id features in train and test set
    id_train = 'id',
    id_test = 'id',

    # name of the trarget feature in the dataset or path to a pickle file
    # or a numpy array 
    # explain about validation 
    target_train = 'target',
    target_test = 'target',

    weights_train = None,
    weights_test = None,

    group_train = None,
    group_test = None,

    n_folds = 5,
    shuffle = True,
)

def create_dataset(splits):
    """
        Create your dataset here !
    """
    train_path = join(params.data_path, 'train', 'train.csv')
    test_path = join(params.data_path, 'test', 'test.csv')
    
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