"""
    class parameters for generate 
"""

from os.path import join
import numpy as np

class ParametersSpace:

    def __init__(self, *args, **kwargs):
        
        # required parameters
        self.project_path  = kwargs.pop('project_path', None)
        self.data_path     = kwargs.pop('data_path', None)
        self.train_name    = kwargs.pop('train_name',None)
        self.target_train  = kwargs.pop('target_train', None)
        self.n_folds       = kwargs.pop('n_folds', None)

        assert self.project_path, "Parameters are not configured correctly"
        assert self.data_path   , "Parameters are not configured correctly"
        assert self.train_name  , "Parameters are not configured correctly"
        assert self.target_train, "Parameters are not configured correctly"
        assert self.n_folds     , "Parameters are not configured correctly"

        # optional parameters with default parameters
        self.seed          = kwargs.pop('seed', 0)
        self.missing       = kwargs.pop('missing', np.nan)

        # optional parameters
        self.test_name     = kwargs.get('test_name')
        self.id_train      = kwargs.get('id_train')
        self.id_test       = kwargs.get('id_test')
        self.target_test   = kwargs.get('target_test')
        self.weights_train = kwargs.get('weights_train')
        self.weights_test  = kwargs.get('weights_test')
        self.groups_train  = kwargs.get('groups_train')
        self.groups_test   = kwargs.get('groups_test')

    def __repr__(self):
        return str(self.__dict__)

    def as_dict(self):
        return self.__dict__