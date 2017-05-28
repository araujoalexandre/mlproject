"""
    utils functions for pickle package
"""
from os.path import isfile, splitext
from pickle import load, dump

def pickle_load(path):
    """
        function to load pickle object
    """
    with open(path, 'rb') as f:
        return load(f)

def _pickle_dump(file, path):
    """
        function to dump picke object
    """
    with open(path, 'wb') as f:
        dump(file, f, -1)

def _get_new_name(path):
    """
        rename file if file already exist
        avoid erasing an existing file
    """
    i = 0
    new_path = path
    while isfile(new_path):
        ext = splitext(path)[1]
        new_path = path.replace(ext, '_{}{}'.format(i, ext))
        i += 1
    return new_path

def pickle_dump(file, path, force=False):
    """
        Helper function to dump a file 
        without deleting an existing one
    """
    if force:
        _pickle_dump(file, path)
    elif not force:
        new_path = _get_new_name(path)
        _pickle_dump(file, new_path)