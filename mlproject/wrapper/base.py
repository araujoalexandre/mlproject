from os import getcwd
from os.path import isfile, join
from datetime import datetime

import numpy as np

from mlproject.utils import get_ext_cls
from mlproject.utils import make_directory, pickle_load


class BaseWrapper:

    def __init__(self, params):
        """
            xxx
        """
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
        cls = get_ext_cls()[self.ext]
        data = cls.load(path)
        return data

    def load_target(self, tr_ix, va_ix):
        """
            Load and return y splits based on validation index
        """
        y = pickle_load(join(self.path, "y.pkl"))
        return y[tr_ix], y[va_ix]

    def load_weights(self, tr_ix, va_ix):
        """
            if weights exists, Load and return weights 
            splits based on validation index 
        """
        if isfile(join(self.path, "weights.pkl")):
            weights = pickle_load(join(self.path, "weights.pkl"))
            return weights[tr_ix], weights[va_ix]
        return None, None

    def load_groups(self, tr_ix, va_ix):
        """
            if weights exists, Load and return weights 
            splits based on validation index 
        """
        # if isfile(join(self.path, "weights.pkl")):
        #     weights = pickle_load(join(self.path, "weights.pkl"))
        #     return weights[tr_ix], weights[va_ix]
        return None, None

    def load_train(self, fold):
        """
            Load and return train & cv set from "Fold_X" folder
        """
        fold_folder = "fold_{}".format(fold)
        path_tr = join(self.path, fold_folder, "X_train.{}".format(self.ext))
        path_va = join(self.path, fold_folder, "X_cv.{}".format(self.ext))
        xtr = self._load(path_tr)
        xva = self._load(path_va)
        return xtr, xva

    def load_test(self):
        """
            Load the test dataset from "Test" folder
        """
        path = join(self.fold_folder, "test", "X_test.{}".format(self.ext))
        # XXX : if model with cmdline don't return dataset but path of dataset 
        if self.ext == 'custom':
            return path
        X_test = self._load(path)
        return X_test