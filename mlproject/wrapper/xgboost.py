
from os.path import join, exists
from contextlib import redirect_stdout
from subprocess import Popen, PIPE, STDOUT
import pandas as pd
import numpy as np
import xgboost as xgb

from mlproject.wrapper.base import BaseWrapper
from mlproject.utils.project import make_directory


# XGB with Python Interface
# XGB with command line

class XGBoostWrapper(BaseWrapper):


    def __init__(self, params):
        self.name = 'XGBoost'
        self.booster = params['booster'].copy()
        # self.predict_opt = params.get('predict_option')
        self.verbose = params.get('verbose', None)
        self.models = []
        super(XGBoostWrapper, self).__init__(params)

    def _train(self, xtr, xva, ytr, yva, fold):
        """
            Function to train a model
        """
        make_directory(self.folder)
        # override label
        xtr.set_label(ytr); xva.set_label(yva)
        self.booster['evals'] = [(xtr, 'train'), (xva, 'cv')]
        self.models += [xgb.train(self.params, xtr, **self.booster)]
        self._features_importance(fold)
        self._dump_txt_model(fold)

    def predict(self, data, fold=None):
        """
            function to make and return prediction
        """
        # predict for train and cv
        if isinstance(fold, int):
            iter_ = self.models[fold].best_ntree_limit
            predictions = self.models[fold].predict(data, ntree_limit=iter_)
        # prediction on test dataset
        # predict with all models from different fold and average predictions
        elif fold is None:
            for i in range(len(self.models)):
                best_ntree_limit = self.models[i].best_ntree_limit
                if i == 0:
                    predictions = self.models[i].predict(data, 
                                                ntree_limit=best_ntree_limit)
                else:
                    predictions += self.models[i].predict(data, 
                                                ntree_limit=best_ntree_limit)
            # average predictions
            predictions /= len(self.models)
        # reshape predicitons if dim == 1
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)        
        return predictions

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
        fmap_name = join(self.path, "features.map")
        get_score = self.models[fold].get_score
        metrics = {
            'weight': get_score(fmap=fmap_name, importance_type='weight'),
            'gain':   get_score(fmap=fmap_name, importance_type='gain'),
            'cover':  get_score(fmap=fmap_name, importance_type='cover'),
        }

        folder = join(self.folder, 'seed_{}'.format(self.seed), "fscore")
        make_directory(folder)

        for key, value in metrics.items():
            df = pd.DataFrame({ 'features': list(value.keys()), 
                                    key: list(value.values())})
            df.sort_values(by=key, ascending=True, inplace=True)
            file = join(folder, "{}_{}_{}.csv".format(self.name, key, fold))
            df[['features', key]].to_csv(file, index=False)

    def _dump_txt_model(self, fold):
        """ 
            make and dump model txt file
        """
        fmap_name = join(self.path, "features.map")
        folder = join(self.folder, "seed_{}".format(self.seed), "dump_models")
        filename = "{}_{}.txt".format(self.name, fold)
        make_directory(folder)
        out = join(folder, filename)
        self.models[fold].dump_model(out, fmap=fmap_name, with_stats=True)

    def save_model(self):
        """
            save model as binary file
        """
        for i in range(len(self.models)):
            folder = join(self.folder, "seed_{}".format(self.seed))
            filename = "model_fold_{}.bin".format(i)
            make_directory(folder)
            self.models[i].save_model(join(folder, filename))

