"""
__file__

    functions.py

__description__

    This file provides various functions.
    
__author__

    Araujo Alexandre < aaraujo001@gmail.com >

"""

import sys, os
from subprocess import Popen, PIPE, STDOUT
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb

from .base import Wrapper
from kaggle.utils.functions import make_directory


class XGBoost(Wrapper):


    def __init__(self, params, paths):        

        self.params_booster = params['booster'].copy()
        self.predict_option = params.get('predict_option')
        self.name = 'XGBoost'
        
        super(XGBoost, self).__init__(params, paths)


    def _create_config_file(self, X_train, X_cv):

        config_path = '{}/config.txt'.format(self.model_folder)
        with open(config_path, 'r') as f:
            f.write("task = train\n")
            for key, value in self.params_booster.items():
                f.write("{} = {}\n".format(key, value))
            f.write('\n')
            for key, value in self.params:
                f.write("{} = {}\n".format(key, value))

            f.write("data = {}".format(X_train))
            f.write("eval[cv] = {}".format(X_cv))
            f.write("save_period = 1")
            f.write("model_dir = {}".format(self.model_folder))


    def train(self, X_train, X_cv, y_train, y_cv):
        """
            Function to train a model
        """

        make_directory(self.model_folder)

        self.params_booster['evals'] = [(X_train, 'train'), (X_cv, 'cv')]
        self.model = xgb.train(self.params, X_train, **self.params_booster)

        self._features_importance(self.fold)
        self._dump_txt_model(self.fold)


    def predict(self, X, cv=False):
        """
            function to make and return prediction
        """
        if self.predict_option is None:
            predict = self.model.predict(X)
        elif isinstance(self.predict_option, str):
            if self.predict_option == 'best_ntree_limit':
                ntree_limit = self.model.best_ntree_limit
            elif ntree_limit == 'best_iteration':
                ntree_limit = self.model.best_iteration
            elif ntree_limit == 'best_score':
                ntree_limit = self.model.best_score
            predict = self.model.predict(X, ntree_limit=ntree_limit)
        elif isinstance(self.predict_option, int):
            predict = self.model.predict(X, ntree_limit=self.predict_option)

        return predict


    def _features_importance(self, fold):
        """
            Make and dump features importance file
            'weight':
                The number of times a feature is used to split the data across 
                all trees. 
            'gain' :
                the average gain of the feature when it is used in trees 
            'cover' :
                the average coverage of the feature when it is used in trees
        """
        fmap_name = "{}/features.map".format(self.folder_path)

        weight = self.model.get_score(fmap=fmap_name, importance_type='weight')
        gain = self.model.get_score(fmap=fmap_name, importance_type='gain')
        cover = self.model.get_score(fmap=fmap_name, importance_type='cover')

        metrics = {
            'weight': weight,
            'gain': gain,
            'cover': cover,
        }

        for key, value in metrics.items():

            df = pd.DataFrame({ 
                        'features': list(value.keys()), 
                        key: list(value.values()), 
                    })

            df.sort_values(by=key, ascending=True, inplace=True)
            args_name = [self.model_folder, self.name, key, self.fold]
            name = "{}/{}_{}_{}.csv".format(*args_name)
            df.to_csv(name, index=False)


    def _dump_txt_model(self, fold):
        """ 
            make and dump model txt file
        """
        fmap_name = "{}/features.map".format(self.folder_path)
        file_name = "{}/{}_{}.txt".format(self.model_folder, self.name, fold)
        self.model.dump_model(file_name, fmap=fmap_name, with_stats=True)


    @property
    def get_model(self):
        """
            xxx
        """
        return self.model