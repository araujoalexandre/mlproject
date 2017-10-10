from os import getcwd, makedirs
from os.path import join, exists
from sys import exit

from mlproject.commands import MlprojectCommand

class Command(MlprojectCommand):

    requires_project = True

    def short_desc(self):
        return "update the scoreboard"

    def add_options(self, parser):
        pass

    def run(self, args):
        """Generate dataset for training"""
        print(args)
        print('run training')