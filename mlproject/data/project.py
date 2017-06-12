"""
    fscipt to generate the dataset
"""
from os import getcwd
from os.path import join
from mlproject.utils import ParametersSpace
from mlproject.utils import project_path

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


################################################################################
#####                      Parameters for the project                      #####
################################################################################
"""
    XXX other comment
"""

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

    # name of the traget feature in the dataset or path to a pickle file
    # or a numpy array 
    # explain about validation
    target_train = 'target',
    target_test = 'target',

    weights_train = None,
    weights_test = None,

    group_train = None,
    group_test = None,

    n_folds = 5,
)


################################################################################
#####                    Function to create the dataset                    #####
################################################################################
"""
    XXX
"""

def create_dataset(splits):
    """
        Create your dataset here !
    """
    train_path = join(params.data_path, 'train', 'train.csv')
    test_path = join(params.data_path, 'test', 'test.csv')
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

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
    suffle = True

    # gkf = GroupKFold(n_splits=fold)
    # split = list(gkf.split(y, y, groups=groups))

    skf = StratifiedKFold(n_splits=nfold, shuffle=suffle, random_state=seed)
    split = list(skf.split(y, y))

    # kf = KFold(n_splits=fold, shuffle=suffle, random_state=seed)
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

def make_submit(id_test, preds, args):
    """
        Create the submission file here !
    """
    df_submit = pd.DataFrame({'id':id_test, 'target': preds})
    file_name = "{}/submit_{}_{}_{:.5f}_0.00000.csv.gz".format(*args)
    df_submit.to_csv(file_name, index=False, compression='gzip')