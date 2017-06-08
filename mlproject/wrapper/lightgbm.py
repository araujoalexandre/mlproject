"""
__file__

    lightgbm.py

__description__

    Microsoft LightGBM Wrapper
    https://github.com/Microsoft/LightGBM
    
__author__

    Araujo Alexandre < aaraujo001@gmail.com >

"""

import os, re, copy
from subprocess import Popen, PIPE
import numpy as np
import pandas as pd

from .base import Wrapper
from kaggle.utils.functions import load_features_name, make_directory


class LightGBM(Wrapper):

    def __init__(self, params, paths):

        self.name = 'LightGBM'
        self.file_ext = 'XXX'

        self.params_booster = params.get('booster')
        self.exec_path = os.environ['LIGHTGBM']
        self.name = "LightGBM"
        self.verbose = params['params'].get('verbose')
        self.predict_increment = 0

        super(LightGBM, self).__init__(params, paths)


    def train(self, X_train, X_cv, y_train, y_cv):
        """
            xxx
        """

        make_directory(self.model_folder)

        args = [self.folder_path, self.fold]
        train_path = "{}/Fold_{}/X_train.libsvm".format(*args)
        cv_path = "{}/Fold_{}/X_cv.libsvm".format(*args)

        self.params['task'] = 'train'
        self.params['data'] = train_path
        self.params['valid'] = cv_path

        out = '{}/LightGBM_model_{}.txt'.format(self.model_folder, self.fold)
        self.params['model_out'] = out

        config = '{}/train.config'.format(self.model_folder)
        with open(config, 'w') as f:
            for key, value in self.params.items():
                f.write('{}={}\n'.format(key, value))

        cmd = [self.exec_path, "config={}".format(config)]
        process = Popen(cmd,stdout=PIPE, bufsize=1).communicate()

        with open(self.params['model_out'], mode='r') as file:
            self.model = file.read()


    def predict(self, X, cv=False):
        """
            xxx
        """

        if self.dataset == 'train':
            args = [self.folder_path, self.fold]
            if self.predict_increment == 0:
                predict_filepath = "{}/Fold_{}/X_train.libsvm".format(*args)
                self.predict_increment += 1
            elif self.predict_increment == 1:
                predict_filepath = "{}/Fold_{}/X_cv.libsvm".format(*args)
                self.predict_increment = 0
        elif self.dataset == 'test':
            predict_filepath = "{}/Test/X_test.libsvm".format(self.folder_path)

        if path is not None:
            predict_filepath = path

        args = [self.model_folder, self.fold]
        input_model = "{}/LightGBM_model_{}.txt".format(*args)
        output_results = "{}/LightGBM_preds_{}.txt".format(*args)

        predict_params = ["task=predict\n",
                          "data={}\n".format(predict_filepath),
                          "input_model={}\n".format(input_model),
                          "output_result={}\n".format(output_results)]

        config = '{}/predict.config'.format(self.model_folder)
        with open(config, 'w') as f:
            f.writelines(predict_params)

        cmd = [self.exec_path, "config={}".format(config)]
        process = Popen(cmd, stdout=PIPE, bufsize=1).communicate()

        prections = np.loadtxt(output_results, dtype=float)
        return prections


    def _feature_importance(self, importance_type='weight'):
        """Get feature importance of each feature.
        Importance type can be defined as:
        'weight' - the number of times a feature is used to split the data
        'gain' - the average gain of the feature when it is used in trees
        'cover' - the average coverage of the feature when it is used in trees
        
        Parameters
        ----------
        importance_type: string
            The type of feature importance
        """
        fmap_name = "{}/features.map".format(self.folder_path)
        features_name = load_features_name(fmap_name)

        match = re.findall("Column_(\d+)=(\d+)", self.model)
        importance = match.values()

        df = pd.DataFrame({ 
                    'features':features_name, 
                    'fscore':importance,
                })

        df.sort_values(by='fscore', ascending=True, inplace=True)
        args_name = [self.model_folder, self.name, self.fold]
        name = "{}/{}_fscore_{}.csv".format(*args_name)
        df.to_csv(name, index=False)

    
    @property
    def get_model(self):
        """
            xxx
        """
        model = copy.copy(self)
        model.dataset = 'test'
        return model