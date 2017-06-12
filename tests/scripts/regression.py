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
    from sklearn.datasets import make_classification
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

    target_train = 'target',
    target_test = 'target',

    weights_train = None,

    group_train = None,
    group_test = None,

    n_folds = 5,
    shuffle = True,
)

def create_dataset(splits):
    """
        Create your dataset here !
    """

    nb_features = 3
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
    df_train['target'] = df_train[0] + 2*df_train[1] + 3*df_train[2]
    df_train['id'] = [x for x in range(len(X_train))]

    df_test = pd.DataFrame(X_test)
    df_test.columns = cols
    df_test['target'] = df_test[0] + 2*df_test[1] + 3*df_test[2]
    df_test['id'] = [x for x in range(len(X_train))]

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