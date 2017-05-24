
'''
__file__

    generate.py

__description__

    File to process, generate and dump a dataset for training
    
__author__

    Araujo Alexandre < aaraujo001@gmail.com >

'''
import os
import argparse
import gc, glob
import string, re

from itertools import combinations, product
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import multiprocessing
from subprocess import Popen, PIPE, STDOUT

from kaggle.helper.GenerateWrapper import GenerateWrapper
from kaggle.utils.functions import pickle_dump, pickle_load
from kaggle.utils.functions import CustomPrint, make_submit
from kaggle.utils.functions import target_transform

import nltk
from nltk.stem.porter import PorterStemmer

from scipy.spatial.distance import euclidean

from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans

from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import LabelEncoder

################################
#####    Globals params    #####
################################

path = '/home/PROJET_FOLDER'


def create_dataset(dataset, skf=None):
    """
        Create your dataset here !
    """
    data_path = '{}/data'.format(path)

    df_train = pd.read_csv('{}/train/original/train.csv'.format(data_path), nrows=20000)
    df_test = None

    return df_train, df_test


def generate_splits(fold, X, y, suffle=True, seed=1234, groups=None):

    ############################
    #####    GroupKFold    #####
    ############################
    # gkf = GroupKFold(n_splits=fold)
    # split = list(gkf.split(X, y, groups=groups))

    #################################
    #####    StratifiedKfold    #####
    #################################
    skf = StratifiedKFold(n_splits=fold, shuffle=suffle, random_state=seed)
    split = list(skf.split(X, y))

    #######################
    #####    KFold    #####
    #######################
    # kf = KFold(n_splits=fold, shuffle=suffle, random_state=seed)
    # split = list(kf.split(X, y))

    return split


def dump(types):

    g = GenerateWrapper(path, 
                        id_name='id',
                        target_name='target', 
                        nan_value=-1,
                        n_folds=5,
                        seed=2016, 
                        skf_path=None,
                        custom_dump=None)

    print_ = CustomPrint(g.log)

    ##############################
    #####    TRAINING SET    #####
    ##############################

    g.dataset = 'train'
    print_.to_print('\nMaking train dataset')
    df_train, df_test = create_dataset(g.dataset)
    print_.to_print('Train dataset Done')

    #############################################
    #####    Target, weights, validation    #####
    #############################################

    # target
    g.y_true = df_train[g.target_name].values

    # weight
    g.weights = None

    # validation
    g.validation = generate_splits(g.n_folds, g.y_true, g.y_true)
    
    ######################
    #####    Loop    #####
    ######################
    for fold, (train_index, cv_index) in enumerate(g.validation):

        df_train = g.cleaning(df_train)
        g.get_infos(df_train)

        g.train_index, g.cv_index = train_index, cv_index
        y_train, y_cv = g.split_target()
        w_train, w_cv = g.split_weights()
        X_train, X_cv = g.split_data(df_train)

        for type_ in types:
            g.dump(X_train, y=y_train, fold=fold, weight=w_train ,type_=type_, name='train')
            g.dump(X_cv, y=y_cv, fold=fold, weight=w_cv ,type_=type_, name='cv')

        message = ('Fold {}\tTrain shape\t[{}|{}]\tCV shape\t[{}|{}]')
        args = [fold, *X_train.shape, *X_cv.shape]
        print_.to_print(message.format(*args))

    # save and generate features map
    g.create_feature_map()

    del df_train, X_train, X_cv
    gc.collect()

    ##########################
    #####    TEST SET    #####
    ##########################

    g.dataset = 'test'
    print_.to_print('Making test dataset')

    if df_test is not None:

        df_test = g.cleaning(df_test)
        g.get_infos(df_test)

        print_.to_print('Test dataset Done')

        for type_ in types:
            g.dump(df_test, type_=type_, name='test')

        print_.to_print('\tTest shape\t[{}|{}]'.format(*df_test.shape))

        # do conformity tests
        g.conformity_test()

    # save infos
    g.save_infos()
    g.copy_script()


if __name__ == '__main__':

    choices = ['xgb', 'npz', 'libsvm', 'pkl', 'libffm', 'custom']
    parser = argparse.ArgumentParser(description='Description of the program')
    parser.add_argument('-t','--type', help='Dataset type(s) to generate', 
                                       nargs='+',
                                       choices=choices,
                                       dest='types',
                                       required=True)
    a = parser.parse_args()
    dump(a.types)

