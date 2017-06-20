import sys
import unittest
import glob
import pickle
from os.path import exists, join, abspath, dirname
from shutil import copyfile
from tempfile import TemporaryDirectory, mkdtemp
from subprocess import Popen, PIPE, call
from itertools import product

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from mlproject.utils import get_ext_cls
from mlproject.utils import ProjectPath


class Datasets:

    def __init__(self):

        self.nb_features = 10
        self.seed = 1234

    def make_binary(self):
        
        cols = ['v{}'.format(x) for x in range(self.nb_features)]

        data_params = {
            'n_samples': 20000,
            'n_features': self.nb_features,
            'n_informative': self.nb_features // 2,
            'random_state': self.seed
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

    def make_regression(self):
        
        cols = ['v{}'.format(x) for x in range(self.nb_features)]

        data_params = {
            'n_samples': 20000,
            'n_features': self.nb_features,
            'n_informative': self.nb_features // 2,
            'random_state': self.seed
        }

        X, y = make_classification(**data_params)
        X_train, X_test = np.array_split(X, 2)
        y_train, y_test = np.array_split(y, 2)

        df_train = pd.DataFrame(X_train)
        df_train.columns = cols
        df_train['target'] = df_train[cols].sum(axis=1)
        df_train['id'] = [x for x in range(len(X_train))]

        df_test = pd.DataFrame(X_test)
        df_test.columns = cols
        df_test['target'] = df_train[cols].sum(axis=1)
        df_test['id'] = [x for x in range(len(X_train))]

        return df_train, df_test

    def make_multiclass(self):
        pass


class BaseTests(unittest.TestCase):

    project_name = "proj123"

    def _execute(self, *args, **kwargs):
        cmd = (sys.executable, "-m", "mlproject.cmdline") + args
        return Popen(cmd, stdout=PIPE, stderr=PIPE, **kwargs).communicate()


class StartPojectTests(BaseTests):

    def test_starproject(self):
        with TemporaryDirectory() as path:

            p = self._execute("startproject", self.project_name, path)
            path = join(path, self.project_name)

            self.assertTrue(exists(join(path, "code").strip()))
            self.assertTrue(exists(join(path, "models")))
            self.assertTrue(exists(join(path, "data")))
            self.assertTrue(exists(join(path, "jupyter")))
            for folder in ["raw", "features", "attributes"]:
                self.assertTrue(exists(join(path, "data", "train", folder)))
                self.assertTrue(exists(join(path, "data", "test", folder)))


class GenerateTest(BaseTests):

    def setUp(self):
        self.dir = TemporaryDirectory()
        # run startproject command
        self._execute('startproject', self.project_name, self.dir.name)
        self.path = join(self.dir.name, self.project_name)
        # generate data
        data = Datasets()
        df_train, df_test = data.make_binary()
        # save data
        train_raw_path = join(self.path, "data", "train", "raw")
        test_raw_path = join(self.path, "data", "test", "raw")
        df_train.to_csv(join(train_raw_path, "train.csv"), index=False)
        df_test.to_csv(join(test_raw_path, "test.csv"), index=False)
        # save attribute data [target / id]
        train_attr_path = join(self.path, "data", "train", "attributes")
        test_attr_path = join(self.path, "data", "test", "attributes")
        for feat in ['target', 'id']:
            out = join(train_attr_path, "{}.pkl".format(feat))
            with open(out, 'wb') as f: 
                pickle.dump(df_train[feat].values, f, -1)
            out = join(test_attr_path, "{}.pkl".format(feat))
            with open(out, 'wb') as f: 
                pickle.dump(df_test[feat].values, f, -1)

    def tearDown(self):
        self.dir.cleanup()

    def test_generate(self):

        extensions = list(get_ext_cls().keys())
        p = self._execute('generate', *extensions, cwd=join(self.path, "code"))
        models_dirs = glob.glob(join(self.path, "models", "**"))
        self.assertTrue(len(models_dirs))

        model = join(self.path, "models", models_dirs[0])
        gen = product(range(5), extensions, ['tr', 'va'])
        for fold, ext, set_ in gen:
            fold_path = join(model, "fold_{}".format(fold))
            self.assertTrue(exists(fold_path))
            files = glob.glob(join(fold_path, "*"))
            self.assertTrue(len(files))
        for ext in extensions:
            self.assertTrue(exists(join(model, 
                    "test", "X_test.{}".format(ext))))


if __name__ == '__main__':
    unittest.main()