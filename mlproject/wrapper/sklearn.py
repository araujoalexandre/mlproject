"""
__file__

    sklearn.py

__description__

    Scikit-Learn algorithms wrapper
    
__author__

    Araujo Alexandre < aaraujo001@gmail.com >

"""
import copy

import numpy as np
import pandas as pd

from mlproject.wrapper import BaseWrapper
from mlproject.utils import load_features_name, make_directory


class SklearnWrapper(BaseWrapper):


    def __init__(self, params, paths):
        """
            xxx
        """
        self.model = params['model']
        self.name = self.model.__class__.__name__
        super(SklearnWrapper, self).__init__(params, paths)

    def train(self, X_train, X_cv, y_train, y_cv):
        """
            Function to train a model
        """

        mask = np.isnan(X_train) | (X_train == np.inf)
        X_train[mask] = -9999

        mask = np.isnan(X_cv) | (X_cv == np.inf)
        X_cv[mask] = -9999

        self.model.set_params(**self.params)      
        if hasattr(self.model, 'n_jobs'):
            self.model.set_params(n_jobs=self.n_jobs)
        self.model.fit(X_train, y_train)
        make_directory(self.model_folder)
        self._features_importance()

    def predict(self, X, cv=False):
        """
            function to make and return prediction
        """
        if hasattr(self.model, 'predict_proba'):
            predict = self.model.predict_proba(X)
        else:
            predict = self.model.predict(X)
        return predict


    def _features_importance(self):
        """
            make and dump features importance file
        """
        if hasattr(self.model, 'feature_importances_'):
            fmap_name = "{}/features.map".format(self.folder_path)
            features_name = load_features_name(fmap_name)
            importance = self.model.feature_importances_

            df = pd.DataFrame({ 
                        'features': features_name, 
                        'fscore': importance,
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
        return copy.copy(self.model)