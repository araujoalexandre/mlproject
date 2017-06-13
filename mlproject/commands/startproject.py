from os import getcwd, makedirs
from os.path import join, exists, abspath
from sys import exit
from datetime import datetime
from pkgutil import get_data

from mlproject.commands import MlprojectCommand
from mlproject.utils import make_directory

# XXX : load default scripts dynamically 
TEMPLATES_SCRIPTS = [
    "project.py",
    "parameters.py",
]

class Command(MlprojectCommand):

    requires_project = False

    def syntax(self):
        return "<project_name> [project_dir]"

    def short_desc(self):
        return "Create new project"

    def add_options(self, parser):
        parser.add_argument("project_name", help="name of the project")
        parser.add_argument("project_dir",  nargs='?', default=getcwd(),
                                 help="directory of the project")

    def _save_file(self, file, path):
        with open(path, "wb") as f:
            f.write(file)

    def run(self, args):
        """
            Create and init folder projet
        """
        project_name = args.project_name
        project_dir = args.project_dir
        project_path = join(project_dir, project_name)

        if not exists(project_path):
            makedirs(project_path)
        else:
            # use Raise instead
            print("Folder already exists")
            exit(1)

        # create config file
        date = datetime.now()
        with open(join(project_path, ".project"), "w") as f:
            f.write("mlproject - {} - {:%Y.%m.%d}\n".format(project_name, date))

        # create dirs in project directory
        for dir_type in ['code', 'jupyter', 'models', 'data']:
            make_directory(join(project_path, dir_type))

        # create dirs in data project directory
        for folder in ['train', 'test']:
            for folder_type in ['binary', 'raw', 'features']:
                dir_name  = join(project_path, 'data', folder, folder_type)
                make_directory(dir_name)

        # copy default python script
        for template in TEMPLATES_SCRIPTS:
            file = get_data('mlproject', join('data', template))
            self._save_file(file, join(project_path, 'code', template))

        print("New project {} created ".format(project_name))
        print("   {}\n".format(project_path))
        print("   Put your datasets in the data folder")
        print("   Update the files bellow in the code folder :")
        for file in TEMPLATES_SCRIPTS:
            print("     - {}".format(file))