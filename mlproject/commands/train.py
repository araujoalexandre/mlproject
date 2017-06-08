from os import getcwd, makedirs
from os.path import join, exists
from sys import exit

from mlproject.commands import MlprojectCommand
from mlproject.api import TrainWrapper
from mlproject.utils import parent_folder
from mlproject.utils import print_and_log as print_


# XXX : print info about size (Go) of model_folder

class Command(MlprojectCommand):

    requires_project = True

    def syntax(self):
        return "<model_dir> [args]"

    def short_desc(self):
        return "run training for model folder"

    def add_options(self, parser):

        # parser.add_option("--train", dest="train", action="store_true",
        #                 help="run training after generation")
        pass

    def _inside_train_folder(self, path):
        if not parent_folder(path) == "models":
            print_(self.logger, "this command needs to be "\
                            "executed from inside a training folder")
            return False
        return True

    def _load_models_wrapper(self):
        from parameters import get_models_wrapper
        from metric import metric
        self.models_wrapper = get_models_wrapper()
        self.metric = metric

    def run(self, args):
        """
            Generate dataset for training
        """
        self.path = getcwd()

        print(args)
        print('run training')
        self._inside_train_folder(self.path)
        self._load_models_wrapper()

        print(self.models_wrapper)

        make_submit = None
        train = TrainWrapper(self.path, self.models_wrapper, self.metric, make_submit)
        train.models_loop(save_model=True)

        # XXX load models_to_run 
        # XXX process all models and load Wrapper

