"""
    BaseAPI class
"""
from os.path import join
from inspect import isfunction
from mlproject.utils import pickle_load


class TransformAPI:

    def __init__(self, target_preprocess):

        self.target_preprocess = target_preprocess
        self._y_train = None
        self._y_test = None

    # @property
    # def y_train(self):
    #     if self._y_train is None:
    #         return None
    #     ret = None
    #     if isfunction(self.target_preprocess):
    #         ret = self.target_preprocess(self._y_train)
    #     if ret is None:
    #         return self._y_train
    #     else:
    #         return ret

    # @property
    # def y_test(self):
    #     if self._y_test is None:
    #         return None
    #     ret = self.target_preprocess(self._y_test)
    #     if ret is None:
    #         return self._y_test
    #     else:
    #         return ret

    # @y_train.setter
    # def y_train(self, value):
    #     self._y_train = value

    # @y_test.setter
    # def y_test(self, value):
    #     self._y_test = value


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
