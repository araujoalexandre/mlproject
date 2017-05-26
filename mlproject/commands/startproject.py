from os import getcwd, makedirs
from os.path import join, exists, abspath
from sys import exit
from datetime import datetime
from pkgutil import get_data

import mlproject
from mlproject.commands import MlprojectCommand
from mlproject.utils import make_directory


TEMPLATES_SCRIPTS = [
    "generate.py.tmpl",
    "parameters.py.tmpl",
    "train.py.tmpl",
]

TEMPLATES_JUPYTER = [
    
]


class Command(MlprojectCommand):

    requires_project = False

    def syntax(self):
        return "<project_name>"

    def short_desc(self):
        return "Create new project"

    def add_options(self, parser):

        parser.add_argument("project_name", 
                            help="choose the name of the project")
        choices = ["regression", "multiclass", "binary"]
        parser.add_argument("--task", choices=choices, default="binary", 
                        help="choose the task of the project, default binary")
        parser.add_argument("--no_test", dest="no_test", action="store_true",
                        help="setup project withouy test set, default false")
        parser.add_argument("--no_submit", dest="no_submit", action="store_true", 
                    help="setup project for without submission, default false")
        # - Does your train set has ids ?
        # - does your test set has ids ?
        # - Do you have a target for test

    def render_template(self, file, args):
        return file

    def _save_file(self, file, path):
        with open(path, "wb") as f:
            f.write(file)

    def run(self, args):
        """
            Create and init folder projet
        """
        path = getcwd()
        project_name = args.project_name

        if not exists(join(path, project_name)):
            makedirs(join(path, project_name))
            path = join(path, project_name)
        else:
            # use Raise instead
            exit("Folder already exists")

        date = datetime.now().strftime(format="%Y.%M.%d")
        with open(join(path, ".project"), "w") as f:
            f.write("mlproject - {} - {}\n".format(project_name, date))

        for dir_type in ['code', 'jupyter', 'models', 'data']:
            make_directory(join(path, dir_type))
        
        folders = ['train', 'test']
        if args.no_test: folders.remove('test')
        path = join(path, )

        for folder in folders:
            for folder_type in ['pkl', 'orginal', 'features']:
                dir_name  = join(path, 'data', folder, folder_type)
                make_directory(dir_name)

        for template in TEMPLATES_SCRIPTS:
            file = get_data('mlproject', join('data', template))
            file = self.render_template(file, args)

            file_name = template[:-5]
            self._save_file(file, join(path, 'code', file_name))

        print("New project {} created ".format(project_name))
        print("   {}\n".format(abspath(project_name)))
        print("   You need to put your datsets in the data folder")
        print("   Modify the generate.py file in the code folder\n")