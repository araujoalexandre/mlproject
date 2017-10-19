from os import getcwd, makedirs
from os.path import join, exists, basename
from sys import exit
from datetime import datetime

import numpy as np

from mlproject.commands import MlprojectCommand
from mlproject.api.generate import GenerateWrapper
from mlproject.utils.serialization import pickle_load
from mlproject.utils.log import print_and_log as print_
from mlproject.utils.project import current_folder, load_project_functions


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

    def run(self, args):
        """Generate dataset for training"""
        self.path = getcwd()

        if not self._inside_project(self.path): return
        if not self._inside_code_folder(self.path): return
        
        functions = load_project_functions()
        GenerateWrapper(
                functions['define_params'](),
                functions['create_dataset'],
                functions['validation_splits'],
                args.extensions
            ).generate_project()