"""
__file__

    base.py

__description__

    This file provides a base class for train algorithm interface.
    
__author__

    Araujo Alexandre < aaraujo001@gmail.com >

"""

import os, datetime

import xgboost as xgb
import pandas as pd
import numpy as np

from sklearn.datasets import load_svmlight_file

from kaggle.utils.functions import make_directory, pickle_load


class Wrapper:

    def __init__(self, params, paths):
        """
            xxx
        """
        self.data_path = paths.get('data_path')
        self.folder_path = paths.get('folder_path')

        self.date = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M")
        self.folder_name = "{}_{}".format(self.name, self.date)
        self.model_folder = "{}/{}".format(self.folder_path, self.folder_name)

        self.params = params['params'].copy()
        self.ext = params['ext']
        self.n_jobs = params.get('n_jobs', -1)

        self.dataset = 'train'
        self.fold = 0


    def __str__(self):
        """
            xxx
        """
        if len(self.params.items()) > 0:
            return str(self.params)
        else:
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
        path = '{}/Fold_{}'.format(self.folder_path, self.fold)
        path_train = '{}/X_train.{}'.format(path, self.ext)
        path_cv = '{}/X_cv.{}'.format(path, self.ext)

        if self.ext == 'custom':
            return path_train, path_cv

        X_train = self._load(path_train)
        X_cv = self._load(path_cv)

        return X_train, X_cv


    def load_target(self):
        """
            Load and return y_train, y_cv from "Fold_X" folder
        """
        path_train = "{}/Fold_{}/y_train.pkl".format(self.folder_path, self.fold)
        path_cv = "{}/Fold_{}/y_cv.pkl".format(self.folder_path, self.fold)
        return pickle_load(path_train), pickle_load(path_cv)


    def load_weights(self):
        """
            if weights exists, load wtrain and wcv
        """
        path_train = "{}/Fold_{}/w_train.pkl".format(self.folder_path, self.fold)
        path_cv = "{}/Fold_{}/w_cv.pkl".format(self.folder_path, self.fold)
        if os.path.isfile(path_train) and os.path.isfile(path_cv):
            return pickle_load(path_train), pickle_load(path_cv)
        return None, None


    def load_test(self):
        """
            Load the test dataset from "Test" folder
        """
        path = "{}/Test/X_test.{}".format(self.folder_path, self.ext)
        
        if self.ext == 'custom':
            return path

        X_test = self._load(path)

        return X_test