
from os.path import join, exists
from contextlib import redirect_stdout
from subprocess import Popen, PIPE, STDOUT

import pandas as pd
import numpy as np
import lightgbm as lgb

from mlproject.wrapper.base import BaseWrapper
from mlproject.utils.project import make_directory
from mlproject.utils.functions import load_features_name


class LightGBMWrapper(BaseWrapper):

    def __init__(self, params):
        self.name = 'LightGBM'
        self.booster = params['booster'].copy()
        self.params = params['params'].copy()
        self.models = []
        super(LightGBMWrapper, self).__init__(params)

    def _train(self, xtr, xva, ytr, yva, fold):
        """
            Function to train a model
        """
        make_directory(self.folder)
        if not isinstance(xtr, lgb.Dataset) or not isinstance(xva, lgb.Dataset):
            xtr = lgb.Dataset(xtr, ytr)
            xva = lgb.Dataset(xva, yva)
        else:
            # override label
            xtr.set_label(ytr); xva.set_label(yva)
        self.models += [lgb.train(self.params, xtr, valid_sets=xtr, **self.booster)]
        self._features_importance(fold)
        self._dump_txt_model(fold)

    def predict(self, data, fold=None):
        """
            function to make and return prediction
        """
        if fold is None:
            prediction = np.zeros((data.num_data(), 0))
            for i in range(len(self.models)):
                gbm = self.models[i]
                best_iter = gbm.best_iteration
                fold_prediction = gbm.predict(data, num_iteration=best_iter)
                prediction = np.hstack((prediction, fold_prediction))
        if isinstance(fold, int):
            gbm = self.models[fold]
            prediction = gbm.predict(data, num_iteration=gbm.best_iteration)
        return prediction

    def _features_importance(self, fold):
        """Get feature importance of each feature.
        importance_type : str, default "split"
            How the importance is calculated: "split" or "gain"
            "split" is the number of times a feature is used in a model
            "gain" is the total gain of splits which use the feature
        """
        feat_names = load_features_name(join(self.path, "features.map"))
        func = self.models[fold].feature_importance
        metrics = {
            'split': func(importance_type='split'),
            'gain': func(importance_type='gain'),
        }

        folder = join(self.folder, 'seed_{}'.format(self.seed), "fscore")
        make_directory(folder)

        for metric, values in metrics.items():
            df = pd.DataFrame({ 'features': feat_names, 
                                  metric: list(values) })
            df.sort_values(by=metric, ascending=True, inplace=True)
            file = join(folder, "{}_{}_{}.csv".format(self.name, metric, fold))
            df.to_csv(file, index=False)

    def _dump_txt_model(self, fold):
        """ 
            make and dump model txt file
        """
        folder = join(self.folder, "seed_{}".format(self.seed), "dump_models")
        filename = "{}_{}.txt".format(self.name, fold)
        make_directory(folder)
        out = join(folder, filename)
        self.models[fold].save_model(out)

    def save_model(self):
        """
            save model as binary file
        """
        pass








