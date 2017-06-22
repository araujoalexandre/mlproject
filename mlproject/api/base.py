"""
    BaseAPI class
"""
from os.path import join
from mlproject.utils import pickle_load


class BaseAPI:

    def _load_target(self):
        """
            load target in data/attributes folder
        """
        attr_path = self.project.data.train.attributes()
        path = join(attr_path, self.params.target_train)
        self.y_train = pickle_load('{}.pkl'.format(path))
        self.y_test = None
        if self.params.target_test is not None:
            attr_path = self.project.data.test.attributes()
            path = join(attr_path, self.params.target_test)
            self.y_test = pickle_load('{}.pkl'.format(path))

    def _load_id(self):
        """
            load id in data/attributes folder
        """
        self.id_train, self.id_test = None, None
        if self.params.id_train is not None:
            attr_path = self.project.data.train.attributes()
            path = join(attr_path, self.params.id_train)
            self.id_train = pickle_load('{}.pkl'.format(path))
        if self.params.id_test is not None:
            attr_path = self.project.data.test.attributes()
            path = join(attr_path, self.params.id_test)
            self.id_test = pickle_load('{}.pkl'.format(path))

    def _load_weights(self):
        """
            load weights in data/attributes folder
        """
        self.weights_train, self.weights_test = None, None
        if self.params.weights_train is not None:
            attr_path = self.project.data.train.attributes()
            path = join(attr_path, self.params.weights_train)
            self.weights_train = pickle_load('{}.pkl'.format(path))
        if self.params.weights_test is not None:
            attr_path = self.project.data.test.attributes()
            path = join(attr_path, self.params.weights_test)
            self.weights_test = pickle_load('{}.pkl'.format(path))

    def _load_groups(self):
        """
            load groups in data/attributes folder
        """
        self.groups_train, self.groups_test = None, None
        if self.params.groups_train is not None:
            attr_path = self.project.data.train.attributes()
            path = join(attr_path, self.params.groups_train)
            self.groups_train = pickle_load('{}.pkl'.format(path))
        if self.params.groups_test is not None:
            attr_path = self.project.data.test.attributes()
            path = join(attr_path, self.params.groups_test)
            self.groups_test = pickle_load('{}.pkl'.format(path))

    def load_attributes(self):
        """
            load attributes from data/attributes folder
        """
        self._load_id()
        self._load_target()
        self._load_weights()
        self._load_groups()