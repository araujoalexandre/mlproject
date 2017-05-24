"""
__file__

    GenerateWrapper.py

__description__

    Class wrapper for generate file
    
__author__

    Araujo Alexandre < aaraujo001@gmail.com >

"""

import os, sys
import argparse
import logging
import random
import datetime
import shutil

import pandas as pd
import numpy as np

from xgboost import DMatrix
from sklearn.datasets import dump_svmlight_file

from kaggle.utils.functions import to_print, pickle_load, pickle_dump


class GenerateWrapper:


    def __init__(   self, 
                    path, 
                    id_name='id', 
                    target_name='target',
                    nan_value=-1, 
                    n_folds=5,
                    seed=1234, 
                    skf_path=None,
                    custom_dump=None):

        self.path = path
        self.data_path = '{}/data'.format(self.path)

        self.id_name = id_name
        self.target_name = target_name
        
        self.nan_value = nan_value
        self.n_folds = n_folds

        self.y_true = None
        self.weights = None
        self.validation = None

        self.train_index = None
        self.cv_index = None

        self.seed = seed

        self.dataset = 'train'

        self.custom_dump = custom_dump

        self.train_shape = (None, None)
        self.test_shape = (None, None)
        
        self.folder_path = self._create_folder()
        self.log = self._init_logger()
        
        self._print_params()


    def _init_logger(self):
        """
            init logger for generate file
        """
        logger = logging.getLogger()
        logging.basicConfig(filename='logs.log',level=logging.INFO)
        return logger


    def _create_folder(self):
        """
            xxx
        """
        timestamp = str(datetime.datetime.now().strftime("%Y.%m.%d_%H.%M"))
        path = "{}/models/model_{}".format(self.path, timestamp)    
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            # use raise instead 
            sys.exit("Folder already exist, wait 1 min")
        return path


    def _print_params(self):
        """
            print global params
        """
        args_msg = [str(datetime.datetime.now().strftime("%Y.%m.%d %H.%M")),
                    self.folder_path.split('/')[-1],
                    self.nan_value, 
                    self.n_folds, 
                    self.seed]
        message = ( "\n### START ###\n\n{}"
                    "\nFolder : {}\n"
                    "\nPARAMS\n"
                    "NAN_VALUE\t{}\n"
                    "FOLDS\t\t{}\n"
                    "SEED\t\t{}"
                )
        to_print(self.log, message.format(*args_msg))


    def save_target(self, y):
        """
            save target in model folder
        """
        y_path = '{}/y_true.pkl'.format(self.folder_path)
        if y is not None and not os.path.isfile(y_path):
            pickle_dump(y, y_path)


    def split_target(self):
        """
            split target array based on train and cv index
        """
        return self.y_true[self.train_index], self.y_true[self.cv_index]


    def split_weights(self):
        """
            split weight array based on train and cv index
        """
        if self.weights is not None:
            return self.weights[self.train_index], self.weights[self.cv_index]
        return None, None


    def split_data(self, df):
        """
            split DataFrame array based on train and cv index
        """
        return df.loc[self.train_index].values, df.loc[self.cv_index].values

    
    def _dump_ffm(self, X, y, path):
        """
            xxx
        """
        ## CHECK IF X IS SPARSE
        ## iterate over rows : x.todense().tolist()[0]

        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X, dtype=np.int)
        with open(path, 'w') as f:
            for i, line in enumerate(X):
                out = str('{} '.format(int(y[i])))
                for col_index, value in enumerate(line):
                    out += '{}:{}:1 '.format(abs(int(col_index)), abs(int(value)))
                out += '\n'
                f.write(out)

        # cols = pickle_load('field_index_ffm.tmp')
        # # X[X == self.nan_value] = np.nan
        # with open(path, 'w') as f:
        #     for i, x in enumerate(X.tolist()):
        #         line = str('{} '.format(y[i]))
        #         filter_list = filter(lambda x: x[1] != 0, zip(cols, x))
        #         line += ' '.join(map(lambda x: '{}:{}'.format(*x),filter_list))
        #         line += '\n'
        #         f.write(line)


    def dump(self, X, y=None, weight=None, fold=None, type_='xgb', name='train'):
        """
            save X, y and weights in the right format
        """

        if fold is not None:
            dump_folder = 'Fold_{}'.format(fold)
        else:
            dump_folder = 'Test'

        # create fold or test folder
        path = "{}/{}".format(self.folder_path, dump_folder)
        if not os.path.exists(path):
            os.makedirs(path)

        y_path = '{}/y_{}.pkl'.format(path, name)
        if y is not None and not os.path.isfile(y_path):
            pickle_dump(y, y_path)

        w_path = '{}/w_{}.pkl'.format(path, name)
        if weight is not None and not os.path.isfile(w_path):
            pickle_dump(weight, w_path)

        X_path = '{}/X_{}.{}'.format(path, name, type_)
        if type_ == 'xgb':
            if y is not None:
                dxgb = DMatrix(X, y, missing=self.nan_value, weight=weight)
            else:
                dxgb = DMatrix(X, missing=self.nan_value, weight=weight)
            dxgb.save_binary(X_path)
        
        elif type_ == 'pkl':
            pickle_dump(X, X_path)

        elif type_ == 'libsvm':
            if y is None:
                y = np.zeros(len(X))
            dump_svmlight_file(X, y, X_path)

        elif type_ == 'npz':
            np.savez_compressed(X_path, X)

        elif type_ == 'libffm':
            if y is None:
                y = np.zeros(len(X))
            self._dump_ffm(X, y, X_path)

        elif type_ == 'custom':
            if y is None:
                y = np.zeros(len(X))
            self.custom_dump(X, y, X_path)


    def cleaning(self, df):
        """
            Remove  target features 
                    id features
            fill nan values
        """
        todrop = []
        if self.target_name in df.columns:
            todrop.append(self.target_name)
        if self.id_name in df.columns:
            todrop.append(self.id_name)        
        df.drop(todrop, axis=1, inplace=True)
        df.fillna(self.nan_value, inplace=True)
        return df


    def get_infos(self, df):
        """
            get infos from dataset
        """
        # save infos
        if self.dataset == 'train':
            self.train_shape = df.shape
            self.train_cols_name = df.columns
        if self.dataset == 'test':
            self.test_shape = df.shape
            self.test_cols_name = df.columns


    def create_feature_map(self):
        """
            function to create features mapping for 
            XGBoost features importance
        """
        outfile = open('{}/features.map'.format(self.folder_path), 'w')
        for i, feat in enumerate(self.train_cols_name):
            outfile.write('{0}\t{1}\tq\n'.format(i, str(feat).replace(' ', '')))
        outfile.close()


    def conformity_test(self):
        """
            xxx
        """
        error = False
        if self.train_shape[1] != self.test_shape[1]:
            message = "shape of DataFrames not equal : train {}, test {}"
            shapes = [self.train_shape[1], self.test_shape[1]]
            to_print(self.log, message.format(*shapes))
            error = True

        if not np.array_equal(self.train_cols_name, self.test_cols_name):
            to_print(self.log, "Columns of dataframes not equal")

        if error:
            a = set(self.train_cols_name)
            b = set(self.test_cols_name)
            diff_cols = a.symmetric_difference(b)
            for name in diff_cols:
                to_print(self.log, "{}".format(name))

    
    def save_infos(self):
        """
            xxx
        """
        to_print(self.log, "Dumping Info")
        infos = {
            'seed': self.seed,
            'data_path': self.data_path,
            'folder_path': self.folder_path,
            'n_folds': self.n_folds,
            'train_shape': self.train_shape,
            'test_shape': self.test_shape,
        }
        
        path = '{}/infos.pkl'.format(self.folder_path)
        pickle_dump(infos, path)

        path = '{}/y_true.pkl'.format(self.folder_path)
        pickle_dump(self.y_true, path)

        if self.weights is not None:
            path = '{}/weights.pkl'.format(self.folder_path)
            pickle_dump(self.weights, path)
        
        path = '{}/validation.pkl'.format(self.folder_path)
        pickle_dump(self.validation, path)


    def copy_script(self):
        """
            xxx
        """
        files = ["generate.py", "train.py", "parameters.py"]
        for script_name in files:
            source = "{}/code/{}".format(self.path, script_name)
            destination = "{}/{}".format(self.folder_path, script_name)
            shutil.copyfile(source, destination)


