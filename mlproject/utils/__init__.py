"""
__file__

    __init__.py

__description__

    The module utils provides different helper functions
    
__author__

    Araujo Alexandre < alexandre.araujo@wavestone.fr >

"""
from .functions import make_submit
from .functions import load_features_name
from .functions import is_pandas
from .functions import is_numpy

from .project import make_directory
from .project import find_project_file
from .project import parent_folder
from .project import current_folder
from .project import inside_project

from .pkl import pickle_load
from .pkl import pickle_dump

from .log import ProgressTable
from .log import print_and_log
from .log import init_log

from .timer import Timer

__all__ = [

    'make_submit',
    'load_features_name',
    'is_pandas',
    'is_numpy',

    'make_directory',
    'find_project_file',
    'parent_folder',
    'current_folder',
    'inside_project',

    'pickle_load',
    'pickle_dump',
    
    'ProgressTable',
    'print_and_log',
    'init_log',

    'Timer',
]