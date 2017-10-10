import copy
from os.path import join

import numpy as np
import pandas as pd

from mlproject.wrapper.base import BaseWrapper
from mlproject.utils.project import make_directory
from mlproject.utils.functions import load_features_name
from mlproject.utils.serialization import pickle_dump


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

    def _predict(self, data, model_id, X):
        """
            function to make and return prediction
        """
        if hasattr(self.model[model_id], 'predict_proba'):
            func_predict = self.model[model_id].predict_proba
        else:
            func_predict = self.model[model_id].predict
        predictions = func_predict(data)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        # if class == 2 => binary classification 
        if predictions.shape[1] == 2:
            predictions = predictions[:, 1].reshape(-1, 1)
        return predictions

    def predict(self, data, fold=None):
        """
            function to make and return prediction
        """
        # predict for train and cv         
        if isinstance(fold, int):
            predictions = self._predict(data, fold)
        # prediction on test dataset
        # predict with all models from different fold and average predictions
        elif fold is None:
            for model_id in range(len(self.models)):
                if model_id == 0:
                    predictions = self._predict(data, model_id) 
                else:
                    predictions += self._predict(data, model_id)
            # average predictions
            predictions /= len(self.models)
        # reshape predicitons if dim == 1
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)        
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