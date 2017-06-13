from os import getcwd
from os.path import exists, join
from datetime import datetime
from contextlib import redirect_stdout

import numpy as np

from mlproject.utils import get_ext_cls
from mlproject.utils import make_directory, pickle_load
from mlproject.utils import background


class BaseWrapper:

    def __init__(self, params):

        self.path = getcwd()
        self.date = datetime.now()
        self.folder = join(self.path, "{}_{:%m%d%H%M}".format(\
                                                        self.name, self.date))

        # XXX : check all params
        self.params = params.pop('params')
        self.ext = params.pop('ext')
        self.n_jobs = params.get('n_jobs', -1)

        self.task = None

        self.y = None
        self.groups = None
        self.weights = None

    def __str__(self):
        if len(self.params.items()) > 0:
            return str(self.params)
        return ''

    def train(self, *args, **kwargs):
        with open(join(self.path, 'verbose_model.log'), 'a') as f:
            with redirect_stdout(f):
                print("\n\n{}".format(self.name))
                print(self.params)
                self._train(*args, **kwargs)

    def _load_data_ext(self, path):
        """
            private function to load a dataset
        """
        cls = get_ext_cls()[self.ext]
        data = cls.load(path)
        return data

    def load_target(self):
        """
            Load and return y splits based on validation index
        """
        return pickle_load(join(self.path, "y.pkl"))

    def split_target(self, tr_ix, va_ix):
        """
            Load and return y splits based on validation index
        """
        y = self.load_target()
        return y[tr_ix], y[va_ix]

    def load_weights(self):
        """
            if weights exists, Load and return weights 
            splits based on validation index 
        """
        if exists(join(self.path, "weights.pkl")):
            return pickle_load(join(self.path, "weights.pkl"))
        return None

    def split_weights(self, tr_ix, va_ix):
        """
            if weights exists, Load and return weights 
            splits based on validation index 
        """
        weights = self.load_weights()
        if weights is not None:
            return weights[tr_ix], weights[va_ix]
        return None, None

    def load_groups(self):
        """
            if weights exists, Load and return weights 
            splits based on validation index 
        """
        if exists(join(self.path, "groups.pkl")):
            return pickle_load(join(self.path, "weights.pkl"))
        return None

    def split_groups(self, tr_ix, va_ix):
        """
            if weights exists, Load and return weights 
            splits based on validation index 
        """
        groups = self.load_groups()
        if groups is not None:
            return groups[tr_ix], groups[va_ix]
        return None, None

    def split_train(self, fold):
        """
            Load and return train & cv set from "Fold_X" folder
        """
        fold_folder = "fold_{}".format(fold)
        path_tr = join(self.path, fold_folder, "X_train.{}".format(self.ext))
        path_va = join(self.path, fold_folder, "X_cv.{}".format(self.ext))
        xtr = self._load_data_ext(path_tr)
        xva = self._load_data_ext(path_va)
        return xtr, xva

    def load_test(self):
        """
            Load the test dataset from "Test" folder
        """
        path = join(self.fold_folder, "test", "X_test.{}".format(self.ext))
        # XXX : if model with cmdline don't return dataset but path of dataset 
        if self.ext == 'custom':
            return path
        X_test = self._load_data_ext(path)
        return X_test