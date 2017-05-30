from os import getcwd, makedirs
from os.path import join, exists, abspath
from sys import exit
from datetime import datetime
from pkgutil import get_data

import mlproject
from mlproject.commands import MlprojectCommand
from mlproject.utils import make_directory


TEMPLATES_SCRIPTS = [
    "dataset.py",
    "parameters.py",
]

TEMPLATES_JUPYTER = [
    
]

# XXX : verify if the path provided is empty folder

class Command(MlprojectCommand):

    requires_project = False

    def syntax(self):
        return "<path/project_name>"

    def short_desc(self):
        return "Create new project"

    def add_options(self, parser):

        parser.add_argument("project_name",
                        help="choose the path or/and the name of the project")

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

        date = datetime.now().strftime(format="%Y.%m.%d")
        with open(join(path, ".project"), "w") as f:
            f.write("mlproject - {} - {}\n".format(project_name, date))

        for dir_type in ['code', 'jupyter', 'models', 'data']:
            make_directory(join(path, dir_type))

        for folder in ['train', 'test']:
            for folder_type in ['binary', 'raw', 'features']:
                dir_name  = join(path, 'data', folder, folder_type)
                make_directory(dir_name)

        for template in TEMPLATES_SCRIPTS:
            file = get_data('mlproject', join('data', template))
            file = self.render_template(file, args)
            self._save_file(file, join(path, 'code', template))

        # XXX : dynamic name files in message 
        print("New project {} created ".format(project_name))
        print("   {}\n".format(abspath(project_name)))
        print("   Put your datasets in the data folder")
        print("   Update the files bellow in the code folder :")
        for file in TEMPLATES_SCRIPTS:
            print("     - {}".format(file))