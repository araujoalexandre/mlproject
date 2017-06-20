from os import getcwd, makedirs
from os.path import join, exists, basename
from sys import exit
from argparse import SUPPRESS
from time import clock
from inspect import isfunction
from importlib import import_module

import numpy as np

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
        """
            convert args from argsparse as class attributes
        """
        self.extensions = args.extensions

    def run(self, args):
        """
            Generate dataset for training
        """
        self.path = getcwd()

        if not self._inside_project(self.path): return
        if not self._inside_code_folder(self.path): return
        
        functions = self._load_functions()
        validation_splits = functions['validation_splits']

        self._extract_args(args)

        define_params = functions.pop("define_params", None)
        create_dataset = functions.pop("create_dataset", None)

        assert define_params, "Project scripts is not define correctly"
        assert create_dataset, "Project scripts is not define correctly"

        if define_params:
            params = define_params()

        gen = GenerateWrapper(params)

        # get logger
        self.logger = gen.logger

        # load attributes
        gen.load_attributes()
        
        # if seed is int convert to list
        if isinstance(params.seed, int):
            seeds = [params.seed]
        else:
            seeds = params.seed

        # loop over seeds values
        for i, seed_value in enumerate(seeds):

            # create validqtion splits
            gen.validation += [validation_splits(   params.n_folds, 
                                                    gen.y_test,
                                                    seed_value
                                                )]
            # Generate df_train & df_test
            print_(self.logger, "\ncreating train/test dataset")
            start = clock()
            df_train, df_test = create_dataset(gen.validation[i])
            print_(self.logger, "train/test set done in {:.0}".format(clock() 
                                                                    - start))

            # XXX : df_train and df_test done
            # what about making the conformity test now 
            # detect nan / inf value and raise error if type format not compatible

            # create folder
            gen.create_folder()
            # clean dataset
            df_train, df_test = gen.cleaning(df_train), gen.cleaning(df_test)
            # save infos
            gen.get_train_infos(df_train)
            gen.get_test_infos(df_test)
            # save and generate features map
            gen.create_feature_map()
            
            # save train fold
            gen._save_train_fold(   self.extensions, 
                                    df_train, 
                                    gen.validation[i], 
                                    seed_value)

            # save test and do conformity tests between train and test
            if df_test is not None:
                gen._save_test(self.extensions, df_test)
                gen.conformity_test()

        # save infos
        gen.save_infos()
        gen.copy_script()