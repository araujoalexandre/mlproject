from os import getcwd, makedirs
from os.path import join, exists, basename
from sys import exit
from argparse import SUPPRESS
from time import clock
from importlib import import_module

import mlproject
from mlproject.commands import MlprojectCommand
from mlproject.api import GenerateWrapper
from mlproject.utils import pickle_load
from mlproject.utils import print_and_log as print_
from mlproject.utils import current_folder

"""
    this file use the GenerateWrapper API 
"""

# XXX : Check if dataset got object features
# XXX : time dataset generation for train and test
# XXX : print info about nan values
# XXX : possiliboty de run the training after the generation process
# XXX : option to clean up after succesfull training

class Command(MlprojectCommand):

    requires_project = True

    def syntax(self):
        return "[formats]"

    def short_desc(self):
        return "generate and save dataset for training"

    def add_options(self, parser):

        choices = ['xgb', 'npz', 'libsvm', 'pkl', 'libffm', 'custom']
        parser.add_argument(dest='extensions', choices=choices, nargs='+',
                            help='Dataset type(s) to generate') 
        parser.add_argument("--train", dest="train", action="store_true",
                        help="run training after generation")

    def _inside_code_folder(self, path):
        if not current_folder(path) == "code":
            print_(self.logger, "this command needs to be "\
                            "executed from the code folder")
            return False
        return True

    def _load_generate_func(self):

        mod = import_module('project')
        self.params = getattr(mod, 'params', None)
        self.create_dataset = getattr(mod, 'create_dataset', None)
        self.validation_splits = getattr(mod, 'validation_splits', None)
        self.metric = getattr(mod, 'metric', None)
        self.make_submit = getattr(mod, 'make_submit', None)

    def _extract_args(self, args):
        """
            convert args from argsparse as class attributes
        """
        self.extensions = args.extensions

    def _try_load_target(self, path):
        """
            try to load target if params.target_train if a path
        """
        if not exists(path):
            return
        if 'pkl' in basename(path):
            try:
                return pickle_load(path)
            except:
                return
        try:
            with open(path, 'r') as f:
                l = []
                while True:
                    val = f.readine()
                    if val == '':
                        break
                    l.append(float(l))
            return l
        except:
            return

    def _save_train_fold(self, gen, df_train, validation):
        """
            save train folds
        """
        for fold, (tr_ix, va_ix) in enumerate(validation):

            ytr, yva = gen.split_target(tr_ix, va_ix)
            wtr, wva = gen.split_weights(tr_ix, va_ix)
            gtr, gva = gen.split_groups(tr_ix, va_ix)
            xtr, xva = gen.split_data(df_train, tr_ix, va_ix)

            kwtr = {'y': ytr, 'weights': wtr, 'groups': gtr, 'fold': fold}
            kwva = {'y': yva, 'weights': wva, 'groups': gva, 'fold': fold}

            for ext in self.extensions:
                gen.dump(xtr, ext, 'tr', **kwtr)
                gen.dump(xva, ext, 'va', **kwva)

            message = ('Fold {}/{}\tTrain shape\t[{}|{}]\tCV shape\t[{}|{}]')
            print_(self.logger, message.format(fold+1, len(validation), 
                                                    *xtr.shape, *xva.shape))

    def _save_test(self, gen, df_test):
        """
            save test set
        """
        # XXX : add target if exists
        # XXX : add weights if exists
        # XXX : add group if exists
        for ext in self.extensions:
            gen.dump(df_test, ext, 'test')
        print_(self.logger, '\t\tTest shape\t[{}|{}]'.format(*df_test.shape))

    def run(self, args):
        """
            Generate dataset for training
        """
        self.path = getcwd()

        if not self._inside_project(self.path): return
        if not self._inside_code_folder(self.path): return
        self._load_generate_func()
        self._extract_args(args)

        params = self.params
        gen = GenerateWrapper(params)

        # get logger
        self.logger = gen.logger

        # XXX no need => only for training step
        # # init pprint class

        # try to load the target 
        gen.y_true = self._try_load_target(params.target_train)
        # if target load successful, we make the validation split
        if gen.y_true:
            gen.validation = self.validation_splits(params.n_folds, gen.y_true)            

        # Generate df_train & df_test
        print_(self.logger, '\ncreating train/test dataset')
        start = clock()
        df_train, df_test = self.create_dataset(gen.validation)
        end = clock()
        print_(self.logger, 'train/test set done in {:.0}'.format(end - start))

        if not gen.y_true:
            # extract target features
            # XXX : add check target_train in df_train.columns
            gen.y_true = df_train[params.target_train].values
            # make validation splits
            gen.validation = self.validation_splits(params.n_folds, gen.y_true)

        # /!\ : LOAD GROUPS and WEIGHTS
        gen.weights = None

        # XXX : df_train and df_test done
        # what about making the conformity test now 
        # detect nan / inf value and raise error if type format not compatible

        # create folder
        gen.create_folder()

        # clean dataset
        df_train = gen.cleaning(df_train)
        df_test = gen.cleaning(df_test)

        # save infos
        gen.get_train_infos(df_train)
        gen.get_test_infos(df_test)

        # save and generate features map
        gen.create_feature_map()

        # save train fold
        self._save_train_fold(gen, df_train, gen.validation)

        # save test and do conformity tests 
        # between train and test
        if df_test is not None:
            self._save_test(gen, df_test)
            gen.conformity_test()

        # save infos
        gen.save_infos()
        gen.copy_script()