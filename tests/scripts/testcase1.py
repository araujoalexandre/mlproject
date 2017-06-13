"""
    testcase 1 : binary classification
"""
from os import getcwd
from os.path import join
from mlproject.utils import ParametersSpace
from mlproject.utils import project_path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

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

        seed = 123456,

        id_train = 'id',
        id_test = 'id',

        target_train = 'target',
        target_test = 'target',

        n_folds = 5,
    )
    return params


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
    from sklearn.datasets import make_classification
    params = define_params()
    nb_features = 10
    cols = ['v{}'.format(x) for x in range(nb_features)]

    data_params = {
        'n_samples': 20000,
        'n_features': nb_features,
        'n_informative': nb_features // 2,
        'random_state': params.seed
    }

    X, y = make_classification(**data_params)
    X_train, X_test = np.array_split(X, 2)
    y_train, y_test = np.array_split(y, 2)

    df_train = pd.DataFrame(X_train)
    df_train.columns = cols
    df_train['target'] = y_train
    df_train['id'] = [x for x in range(len(X_train))]

    df_test = pd.DataFrame(X_test)
    df_test.columns = cols
    df_test['target'] = y_test
    df_test['id'] = [x for x in range(len(X_train))]

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

    # gkf = GroupKFold(n_splits=fold)
    # split = list(gkf.split(y, y, groups=groups))

    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
    split = list(skf.split(y, y))

    # kf = KFold(n_splits=fold, shuffle=True, random_state=seed)
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

def make_submit(path, id_test, yhat, model_name, date, score, args):
    """
        Create the submission file here !
    """
    df_submit = pd.DataFrame({'id':id_test, 'target': yhat})
    file_name = "{}/submit_{}_{}_{:.5f}_0.00000.csv.gz".format(*args)
    df_submit.to_csv(file_name, index=False, compression='gzip')

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