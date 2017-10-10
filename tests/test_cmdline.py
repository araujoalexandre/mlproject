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

from mlproject.utils.functions import get_ext_cls
from mlproject.utils.project import ProjectPath
from mlproject.utils.unit_test import make_binary_dataset


class BaseTests(unittest.TestCase):

    project_name = "project_unit_test"

    def _execute(self, *args, **kwargs):
        cmd = (sys.executable, "-m", "mlproject.cmdline") + args
        return Popen(cmd, stdout=PIPE, stderr=PIPE, **kwargs).communicate()


class StartPojectTests(BaseTests):

    def test_starproject(self):
        with TemporaryDirectory() as path:
            p = self._execute("startproject", self.project_name, path)
            # print erreur
            if p[1] != '': print(p[1].decode('utf8'))
            path = join(path, self.project_name)
            self.assertTrue(exists(join(path, "code").strip()))
            self.assertTrue(exists(join(path, "models")))
            self.assertTrue(exists(join(path, "data")))
            self.assertTrue(exists(join(path, "jupyter")))
            for folder in ["raw", "features", "attributes"]:
                self.assertTrue(exists(join(path, "data", "train", folder)))
                self.assertTrue(exists(join(path, "data", "test", folder)))

class GenerateClassificationTest(BaseTests):

    def setUp(self):
        self.dir = TemporaryDirectory()
        # run startproject command
        self._execute('startproject', self.project_name, self.dir.name)
        self.path = join(self.dir.name, self.project_name)
        make_binary_dataset(self.path)

    def tearDown(self):
        self.dir.cleanup()

    def test_generate(self):
        extensions = list(get_ext_cls().keys())
        p = self._execute('generate', *extensions, cwd=join(self.path, 'code'))
        # print erreur
        if p[1] != '': print(p[1].decode('utf8'))
        models_dirs = glob.glob(join(self.path, 'models', '**'))
        self.assertTrue(len(models_dirs))

        model_path = join(self.path, 'models', models_dirs[0])
        for fold, ext, set_ in product(range(5), extensions, ['tr', 'va']):
            fold_path = join(model_path, 'fold_{}'.format(fold))
            self.assertTrue(exists(fold_path))
            files = glob.glob(join(fold_path, '*'))
            self.assertTrue(len(files))
        for ext in extensions:
            self.assertTrue(exists(join(model_path, 
                'test', 'X_test.{}'.format(ext))))


class XGBClassificationTest(BaseTests):

    def setUp(self):
        self.dir = TemporaryDirectory()
        # run startproject command
        self._execute('startproject', self.project_name, self.dir.name)
        self.path = join(self.dir.name, self.project_name)
        make_binary_dataset(self.path)

    def tearDown(self):
        self.dir.cleanup()

    def test_run_ml(self):
        extensions = list(get_ext_cls().keys())
        # generate dataset : model folder
        p = self._execute('generate', *extensions, cwd=join(self.path, 'code'))
        # print erreur
        if p[1] != '': print(p[1].decode('utf8'))
        models_dirs = glob.glob(join(self.path, 'models', '**'))
        self.assertTrue(len(models_dirs))
        model_folder = models_dirs[0]
        
        # override parameters
        params = (
            "from mlproject.wrapper.xgboost import XGBoostWrapper\n"
            "def get_models_wrapper():\n"
            "    params = {\n"
            "        'ext': 'xgb','predict_option': 'best_ntree_limit',\n"
            "        'booster': {'num_boost_round': 30},\n"
            "        'params': {\n"
            "            'objective': 'binary:logistic',\n"
            "            'eval_metric': 'logloss',\n"
            "            'eta': 0.1,\n"
            "            'silent': 1,\n"
            "        },\n"
            "    }\n"
            "    return [XGBoostWrapper(params)]\n")

        with open(join(model_folder, 'parameters.py'), 'w') as f:
            f.write(params)

        # run ml
        p = self._execute('train', cwd=model_folder)
        # print erreur
        if p[1] != '': print(p[1].decode('utf8'))

        self.assertTrue(exists(join(model_folder, 'logs.log')))
        self.assertTrue(exists(join(model_folder, 'verbose_model.log')))

        # assertion
        output_folder = glob.glob(join(model_folder, 'XGB*'))[0]
        self.assertTrue(exists(join(output_folder, 'train_stack.pkl')))
        self.assertTrue(exists(join(output_folder, 'test_stack.pkl')))

        # check score
        submit_file = glob.glob(join(output_folder, 'submit*'))[0]
        *_, va_score, test_score = submit_file.split('_')
        va_score = float(va_score)
        test_score = float(test_score.replace('.csv.gz', ''))
        # print(va_score, test_score)
        self.assertAlmostEqual(va_score, test_score, places=2)



if __name__ == '__main__':
    unittest.main()