"""
__file__

    api/generate.py

__description__

    Class wrapper for generate file
    
__author__

    Araujo Alexandre < aaraujo001@gmail.com >

"""
from os import makedirs
from os.path import join, isfile
from shutil import copyfile
from datetime import datetime

import numpy as np

from mlproject.utils import is_pandas, is_numpy
from mlproject.utils import print_and_log, init_log
from mlproject.utils import pickle_load, pickle_dump
from mlproject.utils import make_directory
from mlproject.utils import get_ext_cls


# XXX : get and save the shapes of train and test files in project folder ??

class GenerateWrapper:

    def __init__(self, params):

        self.params = params
        self.path = self.params.project_path

        self.train_index, self.cv_index = None, None
        self.train_shape, self.test_shape = (None, None), (None, None)

        self.date = datetime.now()
        self.folder_name = "models_{date:%Y.%m.%d_%H.%M}".format(date=self.date)
        self.folder_path = join(self.path, "models", self.folder_name)

        self.validation = None

        # self._save_custom = None

        self.logger = init_log(self.path)
        self._print_params()

    def create_folder(self):
        """
            function to make the folder
        """
        path = join(self.params.project_path, "models", self.folder_name)
        make_directory(path)

    def _print_params(self):
        """
            print global params
        """
        args_msg = {
            'folder': self.folder_name,
            'date': self.date,
            'missing': self.params.missing,
            'seed': self.params.seed,
            'n_folds': self.params.n_folds,
        }
        message = ( "\n### START ###\n\n{date:%Y.%m.%d %H.%M}"
                    "\nFolder : {folder}\n"
                    "\nPARAMS\n"
                    "NAN_VALUE\t{missing}\n"
                    "FOLDS\t\t{n_folds}\n"
                    "SEED\t\t{seed}"
                )
        print_and_log(self.logger, message.format(**args_msg))

    def split_target(self, tr_index, cv_index):
        """
            split target array based on train and cv index
        """
        # XXX : add check over self.train_index and self.cv_index
        # XXX : add check over self.y_true => dim

        return self.y_true[tr_index], self.y_true[cv_index]

    def split_weights(self, tr_index, cv_index):
        """
            split weight array based on train and cv index
        """
        # XXX : add check over self.train_index and self.cv_index
        # XXX : add check over self.weights => dim
                    
        if self.weights is not None:
            return self.weights[tr_index], self.weights[cv_index]
        return None, None

    def split_groups(self, tr_index, cv_index):
        """
            split groups array based on train and cv index
        """
        return None, None

    def split_data(self, df, tr_index, cv_index):
        """
            split DataFrame array based on train and cv index
        """
        # XXX : add check over self.train_index, self.cv_index
        # XXX : add check over input DataFrame or Numpy array and dim
        # XXX : handle absolute index and real index with loc / iloc method

        return df.loc[tr_index].values, df.loc[cv_index].values

    def dump(self, X, ext, name, y=None, weights=None, groups=None, fold=None):
        """
            save X, y and weights in the right format
        """
        assert is_pandas(X) or is_numpy(X), "dataset need to be Pandas "\
                                                "DataFrame or NumPy array"

        if fold is not None:
            dump_folder = 'fold_{}'.format(fold)
        else:
            dump_folder = 'test'

        # create fold_* or test folder
        path = join(self.folder_path, dump_folder)
        make_directory(path)

        path = join(path, 'X_{}.{}'.format(name, ext))
        cls = get_ext_cls()[ext]
        cls.save(path, X, y, weights=weights, 
                             groups=groups, 
                             missing=self.params.missing)

    def cleaning(self, df):
        """
            Remove  target features 
                    id features
            fill nan values
        """

        # remove space in columns names and convert to str
        features = list(df.columns)
        for i, feat in enumerate(features):
            features[i] = str(feat).replace(' ', '')
        df.columns = features

        # drop ids and target columns
        todrop = []
        checks = [
            self.params.target_train,
            self.params.target_test,
            self.params.id_train,
            self.params.id_test,
        ]
        for col in checks:
            if col and col in df.columns:
                todrop.append(col)
        df.drop(todrop, axis=1, inplace=True)
        
        # fillna
        df.fillna(self.params.missing, inplace=True)
        
        return df

    def get_train_infos(self, df):
        """
            get infos for train set
        """
        self.train_shape = df.shape
        self.train_cols_name = df.columns

    def get_test_infos(self, df):
        """
            get infos for test set
        """
        self.test_shape = df.shape
        self.test_cols_name = df.columns

    def create_feature_map(self):
        """
            function to create features mapping for 
            XGBoost features importance
        """
        with open(join(self.folder_path, "features.map"), 'w') as f:
            for i, feat in enumerate(self.train_cols_name):
                f.write('{0}\t{1}\tq\n'.format(i, feat))

    def conformity_test(self):
        """
            xxx
        """
        error = False

        if self.train_shape[1] != self.test_shape[1]:
            message = "shape of DataFrames not equal : train {}, test {}"
            shapes = [self.train_shape[1], self.test_shape[1]]
            print_and_log(self.logger, message.format(*shapes))
            error = True

        if not np.array_equal(self.train_cols_name, self.test_cols_name):
            print_and_log(self.logger, "Columns of dataframes not equal")

        if error:
            a = set(self.train_cols_name)
            b = set(self.test_cols_name)
            diff_cols = a.symmetric_difference(b)
            for name in diff_cols:
                print_and_log(self.logger, "{}".format(name))
    
    def save_infos(self):
        """
            save infos and files for training step
        """
        print_and_log(self.logger, "Dumping Info")
        infos = {
            "train_shape": self.train_shape,
            "test_shape": self.test_shape,
        }
        
        path = join(self.folder_path, "infos.pkl")
        pickle_dump(infos, path)

        path = join(self.folder_path, "y.pkl")
        pickle_dump(self.y_true, path)

        ## XXX : WEIGHTS / GROUPS

        if self.weights is not None:
            path = '{}/weights.pkl'.format(self.folder_path)
            pickle_dump(self.weights, path)
        
        path = join(self.folder_path, "validation.pkl")
        pickle_dump(self.validation, path)

    def copy_script(self):
        """
            backup dataset.py in folder model
        """
        # XXX load thoses files dynamically
        for script_name in ["project.py", "parameters.py"]:
            source = join(self.path, "code", script_name)
            destination = join(self.folder_path, script_name)
            copyfile(source, destination)