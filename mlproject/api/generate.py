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

from mlproject.api.base import BaseAPI
from mlproject.utils.functions import get_ext_cls, counter
from mlproject.utils.functions import is_pandas, is_numpy
from mlproject.utils.project import ProjectPath, make_directory
from mlproject.utils.log import print_and_log as print_, init_log
from mlproject.utils.serialization import pickle_load, pickle_dump
from mlproject.utils.functions import format_timedelta


class GenerateWrapper(BaseAPI):

    def __init__(self, parameters, functions, extensions):

        self.functions = functions
        self.params = parameters
        self.extensions = extensions

        self.project = ProjectPath(parameters.project_path)

        # self.train_index, self.cv_index = None, None
        self.train_shape, self.test_shape = (None, None), (None, None)

        self.date = datetime.now()
        self.folder_name = "models_{:%Y.%m.%d_%H.%M}".format(self.date)
        self.folder_path = join(self.project.models(), self.folder_name)

        self.validation = []

        # load attributes
        self._load_attributes()

        self.logger = init_log(getcwd())
        self._print_params()

    def _create_folder(self):
        """function to make the folder"""
        path = join(self.project.models(), self.folder_name)
        make_directory(path)

    def _print_params(self):
        """print global params"""
        args_msg = {
            'folder': self.folder_name,
            'date': self.date,
            'missing': self.params.missing,
            'seeds': self.params.seeds,
            'n_folds': self.params.n_folds,
        }
        message = (
            "\n### START ###\n\n{date:%Y.%m.%d %H.%M}"
            "\nFolder : {folder}\n"
            "\nPARAMS\n"
            "NAN_VALUE\t{missing}\n"
            "FOLDS\t\t{n_folds}\n"
            "SEED\t\t{seeds}"
            )
        print_(self.logger, message.format(**args_msg))

    def _split_target(self, tr_index, cv_index):
        """split target array based on train and cv index"""
        # XXX : add check over self.train_index and self.cv_index
        # XXX : add check over self.y_true => dim
        return self.y_train[tr_index], self.y_train[cv_index]

    def _split_weights(self, tr_index, cv_index):
        """split weight array based on train and cv index"""
        # XXX : add check over self.train_index and self.cv_index
        # XXX : add check over self.weights => dim
        if self.weights_train is not None:
            return self.weights_train[tr_index], self.weights_train[cv_index]
        return None, None

    def _split_groups(self, tr_index, cv_index):
        """split groups array based on train and cv index"""
        if self.groups_train is not None:
            gtr, gva = self.groups_train[tr_index], self.groups_train[cv_index]
            return counter(gtr), counter(gva)
        return None, None

    def _split_data(self, df, tr_index, cv_index):
        """split DataFrame array based on train and cv index"""
        # XXX : add check over self.train_index, self.cv_index
        # XXX : add check over input DataFrame or Numpy array and dim
        # XXX : handle absolute index and real index with loc / iloc method
        return df.loc[tr_index].values, df.loc[cv_index].values

    def _dump(self, X, name, ext, y=None, weights=None, groups=None,
              fold=None):
        """save X, y and weights in the right format"""
        if fold is not None:
            dump_folder = 'fold_{}'.format(fold)
        else:
            dump_folder = 'test'

        # create fold_* or test folder
        path = join(self.folder_path, dump_folder)
        make_directory(path)

        path = join(path, 'X_{}.{}'.format(name, ext))
        cls = get_ext_cls()[ext]
        cls.save(
            path, X, y,
            weights=weights,
            groups=groups,
            missing=self.params.missing)

    def _save_train_fold(self, extensions, df_train, validation, seed_value):
        """save train folds"""
        for fold, (tr_ix, va_ix) in enumerate(validation):
            # split all data
            ytr, yva = self._split_target(tr_ix, va_ix)
            wtr, wva = self._split_weights(tr_ix, va_ix)
            gtr, gva = self._split_groups(tr_ix, va_ix)
            xtr, xva = self._split_data(df_train, tr_ix, va_ix)
            # dump all extensions
            for ext in extensions:
                self._dump(xtr, 'tr_{}'.format(seed_value), ext, y=ytr, 
                    weights=wtr, groups=gtr, fold=fold)
                self._dump(xva, 'va_{}'.format(seed_value), ext, y=yva, 
                    weights=wva, groups=gva, fold=fold)
            # print message
            message = ('Fold {}/{}\tTrain shape\t[{}|{}]\tCV shape\t[{}|{}]')
            print_(self.logger, message.format(fold+1, len(validation),
                                            *xtr.shape, *xva.shape))

    def _save_test(self, extensions, df_test, seed_value):
        """save test set"""
        for ext in extensions:
            self._dump(df_test.values, 'test', ext, y=self.y_test, 
                weights=self.weights_test, groups=self.groups_test, fold=None)
        print_(self.logger, '\t\tTest shape\t[{}|{}]'.format(*df_test.shape))

    def _cleaning(self, df):
        """Remove target, id, group and weights features"""
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
        return df

    def _get_infos(self, dataset, df):
        """get infos from DataFrame"""
        setattr(self, '{}_shape'.format(dataset), df.shape)
        setattr(self, '{}_cols_name'.format(dataset), list(df.columns))

    def _create_feature_map(self):
        """create features mapping for XGBoost features importance"""
        with open(join(self.folder_path, "features.map"), 'w') as f:
            for i, feat in enumerate(self.train_cols_name):
                f.write('{0}\t{1}\tq\n'.format(i, feat))

    def _conformity_test(self):
        """check if columns match between train and test"""
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

    def _save_infos(self):
        """save infos and files for training step"""
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

    def _copy_script(self):
        """backup dataset.py in folder model"""
        # XXX load thoses files dynamically
        for script_name in ["project.py", "parameters.py"]:
            source = join(self.project.code(), script_name)
            destination = join(self.folder_path, script_name)
            copyfile(source, destination)

    def create_dataset(self, validation_index):
        """execute create_dataset function from project.py"""
        # Generate df_train & df_test
        print_(self.logger, "\ncreating train/test dataset")
        start = datetime.now()
        df_train, df_test = self.functions['create_dataset'](validation_index)
        print_(self.logger, 
            ("train/test set done in {hours:02d}:{minutes:02d}:{seconds:02d}"
                ).format(**format_timedelta(datetime.now() - start)))

        # save infos
        self._get_infos('train', df_train)
        self._get_infos('test', df_test)

        # clean dataset
        df_train = self._cleaning(df_train)
        df_test = self._cleaning(df_test)

        # conformity tests between train and test before dumping train
        if df_test is not None:
            self._conformity_test()

        return df_train, df_test

    def generate_project(self):
        """loop over seeds values, create validation splits and 
        execute feature engineerring from project.py"""

        # loop over seeds values
        for i, seed_value in enumerate(self.params.seeds):

            # create validation splits
            self.validation += [self.functions['validation_splits'](
                                            self.params.n_folds, 
                                            self.y_train,
                                            seed_value,
                                            self.groups_train
                                            )]

            df_train, df_test = self.create_dataset(self.validation[-1])

            # create folder
            self._create_folder()

            # save and generate features map
            self._create_feature_map()
            
            # save train fold
            self._save_train_fold(   
                self.extensions, 
                df_train, 
                self.validation[-1], 
                seed_value)

            # save test 
            if df_test is not None:
                self._save_test(self.extensions, df_test, seed_value)

        # save infos
        self._save_infos()
        self._copy_script()