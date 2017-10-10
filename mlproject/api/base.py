"""
    BaseAPI class
"""
from os.path import join
from inspect import isfunction

from sklearn.utils.multiclass import type_of_target

from mlproject.utils.serialization import pickle_load
from mlproject.utils.functions import counter

class BaseAPI:

    def _load_target(self):
        """load target in data/attributes folder"""
        path = join(self.attr_path['train'], self.params.target_train)
        self.y_train = pickle_load('{}.pkl'.format(path))
        self.y_test = None
        if self.params.target_test is not None:
            path = join(self.attr_path['test'], self.params.target_test)
            self.y_test = pickle_load('{}.pkl'.format(path))

        # check target
        target_types = set([
            'continuous',
            'binary',
            'multiclass',
            'multiclass-multioutput',
            'continuous-multioutput',
            'multilabel-indicator',
            ])
        self.y_type = type_of_target(self.y_train)
        assert self.y_type in target_types, "target provided not known"
        if self.y_type == 'binary':
            self.n_class = 1
        elif self.y_type in ['multiclass', 'multiclass-multioutput']:
            self.n_class = len(counter(self.y_train))

    def _load_attr(self, name):
        """load attribute in data/attributes folder"""
        # load files if they exist
        for dataset in ['train', 'test']: 
            # define class attribute name
            attr_name = '{}_{}'.format(name, dataset)
            # set class attributes to None
            setattr(self, attr_name, None)
            # get path from params
            path_attr = getattr(self.params, attr_name)
            # override class attribute from before with datafile
            if path_attr is not None:
                load_path = join(self.attr_path[dataset], path_attr)
                file_name = '{}.pkl'.format(load_path)
                setattr(self, attr_name, pickle_load(file_name))

    def _load_attributes(self):
        """load attributes from data/attributes folder"""

        # load path to project data/*/attributes folder
        self.attr_path = {}
        self.attr_path['train'] = self.project.data.train.attributes()
        self.attr_path['test'] = self.project.data.test.attributes()

        # load target for train set and test set if exist
        self._load_target()

        # load id, weights and group file for train and test
        for file in ['id', 'weights', 'groups']:
            self._load_attr(file)