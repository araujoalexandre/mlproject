from os import getcwd, makedirs
from os.path import join, exists
from sys import exit

import mlproject
from mlproject.commands import MlprojectCommand

class Command(MlprojectCommand):

    requires_project = True

    def syntax(self):
        return "<model_dir> [args]"

    def short_desc(self):
        return "fetch generate file from model directory"

    def add_options(self, parser):

        parser.add_argument("models_dir",
                    help="choose the model dir froml which to fetch generate.py")
        parser.add_argument("--best", action="store_false", 
                    help="fetch generate from best model")
        parser.add_argument("--last", dest="no_test", action="store_true",
                    help="setup project withouy test set")

    def run(self, args):
        """
            Fetch generate file from model directory
        """
        print(args)
        print('run fetch')