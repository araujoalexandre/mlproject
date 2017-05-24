"""
__file__

    functions.py

__description__

    This file provides various functions.
    
__author__

    Araujo Alexandre < alexandre.araujo@wavestone.fr >

"""

import os, sys
import pickle
import datetime
import pandas as pd
import numpy as np


def make_directory(path):
    """
        xxx
    """
    if not os.path.exists(path):
        os.makedirs(path)


def pickle_load(path):
    """
        xxx
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def _pickle_dump(file, path):
    """
        xxx
    """
    with open(path, 'wb') as f:
        pickle.dump(file, f, -1)


def _get_new_name(path):
    """
        rename file if file already exist
        avoid erasing an existing file
    """
    i = 0
    new_path = path
    while os.path.isfile(new_path):
        ext = os.path.splitext(path)[1]
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


def to_print(logger, string):
    """
        print and log string
    """
    print(string)
    logger.info(string)


def make_submit(path, name, id_test, preds, score, date, header='id,loss'):
    """
        Function to create sumbit file for Kaggle competition
    """
    ID, TARGET = header.split(',')
    df_submit = pd.DataFrame({ID: id_test, TARGET: preds})
    args = [path, name, date, score]
    file_name = "{}/submit_{}_{}_{:.5f}_0.00000.csv.gz".format(*args)
    df_submit.to_csv(file_name, index=False, compression='gzip')

   
def load_features_name(path):
    """
        xxx
    """
    features_name = []
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            features_name.append(line.split('\t')[1])
    return features_name