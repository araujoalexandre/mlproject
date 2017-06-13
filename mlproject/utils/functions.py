"""
    utils functions ...
"""
import os
import sys
import inspect
from importlib import import_module

import pandas as pd
import numpy as np

from mlproject.utils.io import BaseIO

# XXX : check if pandas is installed otherwise create dummy pandas class

def make_submit(path, name, id_test, yhat, score, date, header=['id','target']):
    """
        Function to create sumbit file for Kaggle competition
    """
    ID, TARGET = header.split(',')
    df_submit = pd.DataFrame({ID: id_test, TARGET: yhat})
    args = [path, name, date, score]
    file_name = "{}/submit_{}_{}_{:.5f}_0.00000.csv.gz".format(*args)
    df_submit.to_csv(file_name, index=False, compression='gzip')
   

def background(func):
    def wrapper(*args, **kwargs):
        backup = sys.stdout
        sys.stdout = None
        if os.fork(): return
        func(*args, **kwargs)
        sys.stdout = backup
        os._exit(os.EX_OK) 
    return wrapper

def load_features_name(path):
    """
        load features map file
    """
    features_name = []
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            features_name.append(line.split('\t')[1])
    return features_name

def is_pandas(df):
    """
        check if df is a pandas DataFrame
    """
    return isinstance(df, pd.DataFrame)

def is_numpy(df):
    """
        check if df is a numpy array
    """
    return isinstance(df, np.ndarray)

def get_ext_cls():
    ext_cls = {}
    mod = import_module('mlproject.utils.io')
    for obj in vars(mod).values():
        if inspect.isclass(obj) and issubclass(obj, BaseIO) \
            and not obj == BaseIO:
                obj_ = obj()
                ext_cls[obj_.ext] = obj_
    return ext_cls