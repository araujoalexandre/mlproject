from os import makedirs, pardir
from os.path import abspath, basename, join, dirname, exists

def make_directory(path):
    """
        check if folder exist, if not create it
    """
    folder_name = current_folder(path)
    assert not exists(path), "{} already exists".format(folder_name)
    makedirs(path)

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
    return basename(abspath(join(path, pardir)))

def current_folder(path):
    return basename(abspath(path))

def inside_project(path):
    return bool(find_project_file(path))