"""
__file__

    api/generate.py

__description__

    Class wrapper for generate file
    
__author__

    Araujo Alexandre < aaraujo001@gmail.com >

"""
from os import makedirs, getcwd
from os.path import join, isfile, exists
from shutil import copyfile
from datetime import datetime

import numpy as np

from mlproject.api import BaseAPI
from mlproject.utils import is_pandas, is_numpy
from mlproject.utils import ProjectPath
from mlproject.utils import print_and_log as print_
from mlproject.utils import init_log
from mlproject.utils import pickle_load, pickle_dump
from mlproject.utils import make_directory
from mlproject.utils import get_ext_cls
from mlproject.utils import counter


# XXX : get and save the shapes of train and test files in project folder ??

class GenerateWrapper(BaseAPI):

    def __init__(self, params):

        self.params = params
        self.project = ProjectPath(params.project_path) 

        # self.train_index, self.cv_index = None, None
        self.train_shape, self.test_shape = (None, None), (None, None)

        self.date = datetime.now()
        self.folder_name = "models_{date:%Y.%m.%d_%H.%M}".format(date=self.date)
        self.folder_path = join(self.project.models(), self.folder_name)

        self.validation = []

        # load attributes
        self.load_attributes()

        self.logger = init_log(getcwd())
        self._print_params()

    def create_folder(self):
        """
            function to make the folder
        """
        path = join(self.project.models(), self.folder_name)
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
        print_(self.logger, message.format(**args_msg))

    def _split_target(self, tr_index, cv_index):
        """
            split target array based on train and cv index
        """
        # XXX : add check over self.train_index and self.cv_index
        # XXX : add check over self.y_true => dim
        return self.y_train[tr_index], self.y_train[cv_index]

    def _split_weights(self, tr_index, cv_index):
        """
            split weight array based on train and cv index
        """
        # XXX : add check over self.train_index and self.cv_index
        # XXX : add check over self.weights => dim
                    
        if self.weights_train is not None:
            return self.weights_train[tr_index], self.weights_train[cv_index]
        return None, None

    def _split_groups(self, tr_index, cv_index):
        """
            split groups array based on train and cv index
        """
        if self.groups_train is not None:
            gtr, gva = self.groups_train[tr_index], self.groups_train[cv_index]
            return counter(gtr), counter(gva)
        return None, None

    def _split_data(self, df, tr_index, cv_index):
        """
            split DataFrame array based on train and cv index
        """
        # XXX : add check over self.train_index, self.cv_index
        # XXX : add check over input DataFrame or Numpy array and dim
        # XXX : handle absolute index and real index with loc / iloc method

        return df.loc[tr_index].values, df.loc[cv_index].values

    def _dump(self, X, name, ext, y=None, weights=None, groups=None, fold=None):
        """
            save X, y and weights in the right format
        """
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

    def _save_train_fold(self, extensions, df_train, validation, seed_value):
        """
            save train folds
        """
        for fold, (tr_ix, va_ix) in enumerate(validation):

            ytr, yva = self._split_target(tr_ix, va_ix)
            wtr, wva = self._split_weights(tr_ix, va_ix)
            gtr, gva = self._split_groups(tr_ix, va_ix)
            xtr, xva = self._split_data(df_train, tr_ix, va_ix)

            kwtr = {'y': ytr, 'weights': wtr, 'groups': gtr, 'fold': fold}
            kwva = {'y': yva, 'weights': wva, 'groups': gva, 'fold': fold}

            for ext in extensions:
                self._dump(xtr, 'tr_{}'.format(seed_value), ext, **kwtr)
                self._dump(xva, 'va_{}'.format(seed_value), ext, **kwva)

            message = ('Fold {}/{}\tTrain shape\t[{}|{}]\tCV shape\t[{}|{}]')
            print_(self.logger, message.format(fold+1, len(validation), 
                                                    *xtr.shape, *xva.shape))

    def _save_test(self, extensions, df_test):
        """
            save test set
        """
        kw = {'y': self.y_test, 'weights': self.weights_test, 
                'groups': self.groups_test, 'fold': None}
        for ext in extensions:
            self._dump(df_test.values, 'test', ext, **kw)
        print_(self.logger, '\t\tTest shape\t[{}|{}]'.format(*df_test.shape))

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
            self.params.weights_train,
            self.params.weights_test,
            self.params.groups_train,
            self.params.groups_test,
        ]
        for col in checks:
            if col and col in df.columns:
                todrop.append(col)
        df.drop(todrop, axis=1, inplace=True)
        
        # fillna
        # df.fillna(self.params.missing, inplace=True)
        
        return df

    def get_train_infos(self, df):
        """
            get infos for train set
        """
        # params = self.params
        self.train_shape = df.shape
        self.train_cols_name = list(df.columns)

        # if params.weights_train is not None and \
        #     params.weights_train in self.train_cols_name:
        #     self.weights_train = df[params.weights_train].values

        # if params.groups_train is not None and \
        #     params.groups_train in self.train_cols_name:
        #     self.groups_train = df[params.groups_train].values

    def get_test_infos(self, df):
        """
            get infos for test set
        """
        # params = self.params
        self.test_shape = df.shape
        self.test_cols_name = list(df.columns)

        # if params.target_test is not None:
        #     self.target_test = self._load_target
        
        # if params.id_test is not None and \
        #     params.id_test in self.test_cols_name:
        #     self.id_test = df[params.id_test].values
        
        # if params.weights_test is not None and \
        #     params.weights_test in self.test_cols_name:
        #     self.weights_test = df[params.weights_test].values

        # if params.groups_test is not None and \
        #     params.groups_test in self.test_cols_name:
        #     self.groups_test = df[params.groups_test].values

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
            message = ("/!\ /!\ /!\ /!\ "
                       "shape of DataFrames not equal : train {}, test {}"
                       "/!\ /!\ /!\ /!\ ")
            shapes = [self.train_shape[1], self.test_shape[1]]
            print_(self.logger, message.format(*shapes))
            error = True

        if not np.array_equal(self.train_cols_name, self.test_cols_name):
            message = ("/!\ /!\ /!\ /!\ "
                       "Columns of dataframes not equal"
                       "/!\ /!\ /!\ /!\ ")
            print_(self.logger, message)
            error = True

        if error:
            a = set(self.train_cols_name)
            b = set(self.test_cols_name)
            diff_cols = a.symmetric_difference(b)
            for name in diff_cols:
                print_(self.logger, "{}".format(name))
    
    def save_infos(self):
        """
            save infos and files for training step
        """
        print_(self.logger, "Dumping Info")
        infos = {
            "train_shape": self.train_shape,
            "test_shape": self.test_shape,
            "project_params": self.params,
        }
        
        path = join(self.folder_path, "infos.pkl")
        pickle_dump(infos, path)

        path = join(self.folder_path, "validation.pkl")
        pickle_dump(self.validation, path)

    def copy_script(self):
        """
            backup dataset.py in folder model
        """
        # XXX load thoses files dynamically
        for script_name in ["project.py", "parameters.py"]:
            source = join(self.project.code(), script_name)
            destination = join(self.folder_path, script_name)
            copyfile(source, destination)