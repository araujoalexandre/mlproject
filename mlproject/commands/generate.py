from os import getcwd, makedirs
from os.path import join, exists
from sys import exit

import mlproject
from mlproject.commands import MlprojectCommand

class Command(MlprojectCommand):

    requires_project = True

    def syntax(self):
        return ""

    def short_desc(self):
        return "generate dataset for training"

    def add_options(self, parser):

        parser.add_option("--train", dest="train", action="store_true",
                        help="run training after generation")

    def run(self, args):
        """
            Generate dataset for training
        """
        print(args)
        print('run generate')










