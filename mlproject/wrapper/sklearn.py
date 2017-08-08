import copy
from os.path import join

import numpy as np
import pandas as pd

from .base import BaseWrapper
from mlproject.utils import load_features_name, make_directory
from mlproject.utils import pickle_dump


class SklearnWrapper(BaseWrapper):


    def __init__(self, params):
        self.model = params['model']
        self.models = []
        self.name = self.model.__class__.__name__
        super(SklearnWrapper, self).__init__(params)

    def train(self, xtr, xva, ytr, yva, fold):
        """
            Function to train a model
        """
        self.models += [copy.copy(self.model)]
        self.models[fold].set_params(**self.params)
        self.models[fold].fit(xtr, ytr)
        make_directory(self.folder)
        self._features_importance(fold)

    def _predict(self, model, X):
        """
            function to make and return prediction
        """
        if hasattr(model, 'predict_proba'):
            func_predict = model.predict_proba
        else:
            func_predict = model.predict
        predictions = func_predict(X)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        # if class == 2 => binary classification 
        if predictions.shape[1] == 2:
            predictions = predictions[:, 1].reshape(-1, 1)
        return predictions

    def predict(self, X, fold=None):
        """
            function to make and return prediction
        """
        if fold is None:
            predictions = np.zeros((len(X), 0))
            for i in range(len(self.models)):
                predictions = np.hstack((predictions, 
                                    self._predict(self.models[i], X)))
        if isinstance(fold, int):
            predictions = self._predict(self.models[fold], X)
        return predictions

    def _features_importance(self, fold):
        """
            make and dump features importance file
        """
        if hasattr(self.models[fold], 'feature_importances_'):
            fmap_name = join(self.path, "features.map")
            names = load_features_name(fmap_name)
            importance = self.models[fold].feature_importances_

            folder = join(self.folder, 'seed_{}'.format(self.seed), "fscore")
            filename = "{}_fscore_{}.csv".format(self.name, fold)
            make_directory(folder)

            df = pd.DataFrame({'features': names, 'fscore': importance})
            df.sort_values(by='fscore', ascending=True, inplace=True)                        
            df.to_csv(join(folder, filename), index=False)

    def save_model(self):
        """
            save model as binary file
        """
        for i in range(len(self.models)):
            folder = join(self.folder, "seed_{}".format(self.seed))
            filename = "model_fold_{}.pkl".format(i)
            make_directory(folder)
            pickle_dump(self.models[i], join(folder, filename))