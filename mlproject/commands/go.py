
from mlproject.commands import MlprojectCommand

class Command(MlprojectCommand):

    requires_project = True

    def syntax(self):
        return "<model_dir>"

    def short_desc(self):
        return "cd to specific model dir"

    def add_options(self, parser):

        parser.add_argument("models_dir",
                    help="choose the model dir to go")

    def run(self, args):
        """
            Fetch generate file from model directory
        """
        print(args)
        print('run go')