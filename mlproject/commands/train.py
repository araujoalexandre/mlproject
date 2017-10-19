from os import getcwd, makedirs
from os.path import join, exists
from sys import exit
from importlib import import_module
from inspect import isfunction

from mlproject.commands import MlprojectCommand
from mlproject.api.train import TrainWrapper
from mlproject.utils.project import parent_folder, load_project_functions
from mlproject.utils.log import print_and_log as print_
from mlproject.utils.functions import background


# XXX : print info about size (Go) of model_folder

class Command(MlprojectCommand):

    requires_project = True

    def syntax(self):
        return "<model_dir> [args]"

    def short_desc(self):
        return "run training for model folder"

    def add_options(self, parser):
        parser.add_argument("--bg", dest="background", action="store_true",
                        help="run training in background")

    def _inside_train_folder(self, path):
        if not parent_folder(path) == "models":
            print("this command needs to be "
                    "executed from inside a training folder")
            return False
        return True

    def _load_wrappers(self):
        mod = import_module('parameters')
        wrappers = mod.get_models_wrapper()
        return wrappers 

    def _run_in_prompt(self, cls_train):
        #XXX : get thoses params from args
        cls_train.train_predict(submit=True, save_model=True)

    @background
    def _run_in_background(self, cls_train):
        #XXX : get thoses params from args
        cls_train.train_predict(submit=True, save_model=True)

    def run(self, args):
        """
            Generate dataset for training
        """
        self.path = getcwd()
        self._inside_train_folder(self.path)
        
        functions = load_project_functions()
        keys = ('create_dataset', 'validation_splits', 'define_params')
        for k in keys:
            del functions[k]

        wrappers = self._load_wrappers()
        cls_train = TrainWrapper(self.path, wrappers, **functions)

        if args.background:
            self._run_in_background(cls_train)
        else:
            self._run_in_prompt(cls_train)
