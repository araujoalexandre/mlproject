from glob import glob
from os import makedirs, pardir
from os.path import abspath, basename, join, dirname, exists, isdir


def make_directory(path):
    """
        check if folder exist, if doesn't exist create it
    """
    if not exists(path): makedirs(path)

def find_project_file(path='.', prevpath=None):
    """
        Return the path to the closest .project file by 
        traversing the current directory and its parents
    """
    if path == prevpath:
        return ''
    path = abspath(path)
    project_file = join(path, '.project')
    if exists(project_file):
        return project_file
    return find_project_file(dirname(path), path)

def parent_folder(path):
    """
        return parent folder of given path
    """
    return basename(abspath(join(path, pardir)))

def current_folder(path):
    """
        return current folder of given path
    """
    return basename(abspath(path))

def inside_project(path):
    """
        return bool if inside project
    """
    return bool(find_project_file(path))

def project_path(path):
    """
        find and return the path of the project
    """
    return dirname(find_project_file(path))


class ProjectPath:

    def __init__(self, path):
        self._path = path
        for path in glob(join(self._path, "**")):
            if isdir(path) and not hasattr(self, basename(path)):
                setattr(self, basename(path), ProjectPath(path))

    def __call__(self):
        return self._path

    def __repr__(self):
        return self._path