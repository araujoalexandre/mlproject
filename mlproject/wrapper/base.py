"""
__file__

    base.py

__description__

    This file provides a base class for train algorithm interface.
    
__author__

    Araujo Alexandre < aaraujo001@gmail.com >

"""
from os.path import isfile, join
from datetime import datetime

import xgboost as xgb
import pandas as pd
import numpy as np

from sklearn.datasets import load_svmlight_file

from mlproject.utils import make_directory, pickle_load


class BaseWrapper:

    def __init__(self, params):
        """
            xxx
        """
        self.date = datetime.now()
        self.folder_name = "{}_{:%m%d%H%M}".format(self.name, self.date)
        self.model_folder = join(self.folder_path, self.folder_name)

        # XXX : check all params
        self.params = params.pop('params')
        self.ext = params.pop('ext')
        self.n_jobs = params.get('n_jobs', -1)

        self.dataset = 'train'
        self.fold = 0

    def __str__(self):
        """
            xxx
        """
        if len(self.params.items()) > 0:
            return str(self.params)
        return ''


    def _load(self, path):
        """
            private function to load a dataset
        """
        if self.ext == 'libsvm':
            X = load_svmlight_file(path)
        elif self.ext == 'pkl':
            X = pickle_load(path)
        elif self.ext == 'xgb':
            X = xgb.DMatrix(path)
        elif self.ext == 'npz':
            with np.load(path) as data:
                X = data['arr_0']
        elif self.ext in 'libffm':
            X = path

        return X


    def load_train(self):
        """
            Load and return train & cv set from "Fold_X" folder
        """
        fold_folder = 'fold_{}'.format(self.fold)
        path_train = join(self.folder_path, fold_folder, "X_tr.pkl")
        path_cv = join(self.folder_path, fold_folder, "X_cv.pkl")

        # XXX : if model with cmdline don't return dataset but path of dataset 
        if self.ext == 'custom':
            return path_train, path_cv
        X_train = self._load(path_train)
        X_cv = self._load(path_cv)
        return X_train, X_cv

    def load_target(self):
        """
            Load and return y_train, y_cv from "Fold_X" folder
        """
        fold_folder = 'fold_{}'.format(self.fold)
        path_train = join(self.folder_path, fold_folder, "y_tr.pkl")
        path_cv = join(self.folder_path, fold_folder, "y_cv.pkl")
        return pickle_load(path_train), pickle_load(path_cv)

    def load_weights(self):
        """
            if weights exists, load wtrain and wcv
        """
        fold_folder = 'fold_{}'.format(self.fold)
        path_train = join(self.folder_path, fold_folder, "w_tr.pkl")
        path_cv = join(self.folder_path, fold_folder, "w_cv.pkl")
        if isfile(path_train) and isfile(path_cv):
            return pickle_load(path_train), pickle_load(path_cv)
        return None, None

    def load_test(self):
        """
            Load the test dataset from "Test" folder
        """
        path = join(self.fold_folder, "test", "X_test.{}".formart)        
        # XXX : if model with cmdline don't return dataset but path of dataset 
        if self.ext == 'custom':
            return path

        X_test = self._load(path)

        return X_test