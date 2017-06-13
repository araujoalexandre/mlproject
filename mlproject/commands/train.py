from os import getcwd, makedirs
from os.path import join, exists
from sys import exit

from mlproject.commands import MlprojectCommand
from mlproject.api import TrainWrapper
from mlproject.utils import parent_folder
from mlproject.utils import print_and_log as print_
from mlproject.utils import background


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
            print("this command needs to be "\
                            "executed from inside a training folder")
            return False
        return True

    def _load_models_wrapper(self):
        from parameters import get_models_wrapper
        from metric import metric
        self.models_wrapper = get_models_wrapper()
        self.metric = metric

    def _run_in_prompt(self, cls_train):
        cls_train.models_loop(save_model=True)

    @background
    def _run_in_background(self, cls_train):
        cls_train.models_loop(save_model=True)

    def run(self, args):
        """
            Generate dataset for training
        """
        self.path = getcwd()
        self._inside_train_folder(self.path)
        self._load_models_wrapper()

        self.make_submit = None

        cls_train = TrainWrapper(   self.path, 
                                    self.models_wrapper, 
                                    self.metric, 
                                    self.make_submit)

        if args.background:
            self._run_in_background(cls_train)
        else:
            # sys.stdout = None
            self._run_in_prompt(cls_train)
