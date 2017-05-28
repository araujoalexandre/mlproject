from os import getcwd, pardir, remove
from os.path import join, exists, dirname, basename, abspath
from sys import exit
from glob import glob

import mlproject
from mlproject.commands import MlprojectCommand
from mlproject.utils import find_project_file

class Command(MlprojectCommand):

    requires_project = True

    def syntax(self):
        return "[model_dirs]"

    def short_desc(self):
        return "clean up model folder"

    def add_options(self, parser):

        parser.add_argument("model_dirs", nargs='*', 
                help="remove dataset from model folder")
        parser.add_argument("--all", action='store_true', 
                help="clean all model foders")

    def _remove_files(self, files):
        if not len(files):
            print("nothing to remove")
            exit(0)

        for file in files:
            print('{}'.format(file))
        confirm = input('remove these files (y/n): ')
        if confirm == "y":
            for file in files:
                remove(file)

    def _find_files(self, path):
        files = glob(join(path, 'fold_*', 'X_*'))
        files += glob(join(path, 'test', 'X_*'))
        return files

    def _run_all(self, args):
        project_file_path = find_project_file(getcwd())
        models_folder = join(dirname(project_file_path), "models")
        files = self._find_files(join(models_folder, "**"))
        self._remove_files(files)

    def _run_infolder(self, args):
        if basename(abspath(join(getcwd(), pardir))) == "models":
            # find files from current dir and remove it
            files = self._find_files(getcwd())
            self._remove_files(files)
        else:
            print("you need to be in a model folder " \
                    "or you need to precise a model folder")

    def _run_specific(self, args):
        project_file_path = find_project_file(getcwd())
        models_folder = join(dirname(project_file_path), "models")
        
        # check all dir in args.model_dirs
        for model_dir in sorted(list(set(args.model_dirs))):
            if not exists(join(models_folder, model_dir)):
                print("{} doesn't exist in models_folder".format(model_dir))
                exit(0)

        files = []
        for model_dir in sorted(list(set(args.model_dirs))):
            if exists(join(models_folder, model_dir)):
                files += self._find_files(join(models_folder, model_dir))
        self._remove_files(files)

    def run(self, args):
        """
            Clean up model folder
        """
        self.check_inside_project(getcwd())

        if args.all:
            self._run_all(args)
        elif not args.all and not args.model_dirs:
            self._run_infolder(args)
        elif not args.all and isinstance(args.model_dirs, list):
            self._run_specific(args)