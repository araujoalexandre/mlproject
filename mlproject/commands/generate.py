from os import getcwd, makedirs
from os.path import join, exists, basename
from sys import exit
from argparse import SUPPRESS
from datetime import datetime
from inspect import isfunction
from importlib import import_module

import numpy as np

from mlproject.commands import MlprojectCommand
from mlproject.api.generate import GenerateWrapper
from mlproject.utils.serialization import pickle_load
from mlproject.utils.log import print_and_log as print_
from mlproject.utils.project import current_folder
from mlproject.utils.functions import format_timedelta

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
        choices = ['xgb', 'lgb', 'npz', 'libsvm', 'pkl', 'libffm']
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

    def _load_functions(self):
        mod = import_module('project')
        functions = {}
        func_list = vars(mod)['load']
        for func in vars(mod).values():
            if isfunction(func) and \
                func.__name__ in func_list:
                functions[func.__name__] = func
        return functions

    def _extract_args(self, args):
        """convert args from argsparse as class attributes"""
        self.extensions = args.extensions

    def run(self, args):
        """Generate dataset for training"""
        self.path = getcwd()

        if not self._inside_project(self.path): return
        if not self._inside_code_folder(self.path): return
        
        functions = self._load_functions()
        validation_splits = functions['validation_splits']

        self._extract_args(args)

        define_params = functions.pop("define_params", None)
        create_dataset = functions.pop("create_dataset", None)

        assert define_params is not None, "project.py is not define correctly"
        assert create_dataset is not None, "project.py is not define correctly"

        gen = GenerateWrapper(define_params())

        # get logger
        self.logger = gen.logger

        # loop over seeds values
        for i, seed_value in enumerate(gen.params.seeds):

            # create validation splits
            gen.validation += [validation_splits(   gen.params.n_folds, 
                                                    gen.y_train,
                                                    seed_value,
                                                    gen.groups_train
                                                )]
            # Generate df_train & df_test
            print_(self.logger, "\ncreating train/test dataset")
            start = datetime.now()
            df_train, df_test = create_dataset(gen.validation[i])
            print_(self.logger, ("train/test set done in "
                "{hours:02d}:{minutes:02d}:{seconds:02d}".format\
                                (**format_timedelta(datetime.now() - start))))

            # XXX : df_train and df_test done
            # what about making the conformity test now 
            # detect nan / inf value and raise error if type format not compatible

            # create folder
            gen.create_folder()
            # clean dataset
            df_train, df_test = gen.cleaning(df_train), gen.cleaning(df_test)

            # save infos
            gen.get_infos('train', df_train)
            gen.get_infos('test', df_test)

            # save and generate features map
            gen.create_feature_map()

            # conformity tests between train and test before dumping train
            if df_test is not None:
                gen.conformity_test()
            
            # save train fold
            gen._save_train_fold(   self.extensions, 
                                    df_train, 
                                    gen.validation[i], 
                                    seed_value)

            # save test 
            if df_test is not None:
                gen._save_test(self.extensions, df_test, seed_value)

        # save infos
        gen.save_infos()
        gen.copy_script()