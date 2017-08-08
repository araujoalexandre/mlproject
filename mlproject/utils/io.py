
"""
    wrapper class for loading and saving 
"""
import os
from os.path import join
from multiprocessing import cpu_count 
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
from xgboost import DMatrix
from lightgbm import Dataset
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from pandas import DataFrame
from .pkl import pickle_dump, pickle_load


class BaseIO:

    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def ext(self):
        return "unknown"

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError


class DMatrixIO(BaseIO):

    @property
    def ext(self):
        return "xgb"

    def save(self, save_path, X, y=None, *args, **kwargs):

        weights = kwargs.get('weights', None)
        groups  = kwargs.get('groups', None)
        missing = kwargs.get('missing', np.nan)
        dmat = DMatrix(X, y, missing=missing, weight=weights, 
                                nthread=cpu_count())
        if groups is not None:
            dmat.set_group(groups)
        dmat.save_binary(save_path)

    def load(self, path):
        dmat = DMatrix(path, silent=True)
        return dmat


class LightGBMIO(BaseIO):

    @property
    def ext(self):
        return "lgb"

    def save(self, save_path, X, y=None, *args, **kwargs):        
        weights = kwargs.get('weights', None)
        groups = kwargs.get('groups', None)
        lgb_dataset = Dataset(X, label=None, weight=weights, group=groups,
                                 silent=True, free_raw_data=False)
        lgb_dataset.save_binary(save_path)

    def load(self, path):
        lgb_dataset = Dataset(path, silent=True).construct()
        return lgb_dataset


class pickleIO(BaseIO):

    @property
    def ext(self):
        return "pkl"

    def save(self, save_path, X, y=None, *args, **kwargs):
        pickle_dump(X, save_path)

    def load(self, path):
        return pickle_load(path)


class npzIO(BaseIO):

    @property
    def ext(self):
        return "npz"

    def save(self, save_path, X, y=None, *args, **kwargs):
        np.savez_compressed(save_path, X)
    
    def load(self, path):
        with np.load(path) as npz:
            data = npz['arr_0']
        return data

# class ffmIO(BaseIO):

#     @property
#     def ext(self):
#         return "ffm"

#     def save(self, save_path, X, y=None, *args, **kwargs):
#         """
#             function to convert and save : libffm format
#             format :
#                 <label> <group> <field>:<index>:<value> <field>:<index>:<value>
#         """
#         if y is None: y = np.zeros(len(X))
#         if PANDAS_INSTALLED and isinstance(X, DataFrame):
#             X = X.values

#         f_abs = lambda x: abs(int(x))
#         # convert all to int ? what about NaN value ?
#         # string io/memory is way faster with a comprehension list
#         # => https://waymoot.org/home/python_string/
#         X = np.array(X, dtype=np.int)
#         with open(save_path, 'w') as f:
#             for i, line in enumerate(X):
#                 out = str('{} '.format(int(y[i])))
#                 for col_index, value in enumerate(line):
#                     out += '{}:{}:1 '.format(f_abs(col_index), f_abs(value))
#                 out += '\n'
#                 f.write(out)

#     def load(self, path):
#         print('load as ffm')


# class svmIO(BaseIO):

#     @property
#     def ext(self):
#         return "svm"

#     def save(self, save_path, X, y=None, *args, **kwargs):
#         """
#             function to convert and save : libsvm format
#             format :
#                 <label> <group> <index>:<value> <index>:<value>
#         """
#         if not SKLEARN_INSTALLED:
#             raise Exception("Package Scikit-Learn needs to be installed")    
#         if y is None: y = np.zeros(len(X))
#         # XXX : remove the dependency to sklearn 
#         # comprehension list is fast 
#         # => https://waymoot.org/home/python_string/
#         dump_svmlight_file(X, y, save_path)

#     def load(self, path):
#         return load_svmlight_file(path)


# class svmgroupIO(BaseIO):

#     @property
#     def ext(self):
#         return "svmgroup"

#     def save(self, save_path, X, y=None, *args, **kwargs):
#         pass

#     def load(self, path):
#         pass





# class csvIO(BaseIO):

#     # @property
#     # def ext(self):
#     #     return "csv"

#     def save(self, save_path, X, y=None, *args, **kwargs):
#         pass

#     def load(self, path):
#         pass


