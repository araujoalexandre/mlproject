
from os.path import join, exists
from contextlib import redirect_stdout
from subprocess import Popen, PIPE, STDOUT
import pandas as pd
import numpy as np
import xgboost as xgb

from .base import BaseWrapper
from mlproject.utils import make_directory


# XGB with Python Interface
# XGB with command line

class XGBoostWrapper(BaseWrapper):


    def __init__(self, params):
        self.name = 'XGBoost'
        self.booster = params['booster'].copy()
        self.predict_opt = params.get('predict_option')
        self.verbose = params.get('verbose', None)
        self.models = []
        super(XGBoostWrapper, self).__init__(params)

    # def _create_config_file(self, X_train_path, X_cv_path):
    #     config_path = join(self.folder, "config.txt")
    #     with open(config_path, 'r') as f:
    #         f.write("task = train\n")
    #         for key, value in self.booster.items():
    #             f.write("{} = {}\n".format(key, value))
    #         f.write('\n')
    #         for key, value in self.params:
    #             f.write("{} = {}\n".format(key, value))
    #         f.write("data = {}".format(X_train_path))
    #         f.write("eval[cv] = {}".format(X_cv_path))
    #         f.write("save_period = 1")
    #         f.write("model_dir = {}".format(self.folder))

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

    def _predict_with_option(self, model_predict, dmat):
        if self.predict_opt is None:
            # ntree_limit = 0
            # best iter as default
            ntree_limit = model_predict.best_ntree_limit
        elif isinstance(self.predict_opt, str):
            ntree_limit = {
                'best_ntree_limit': model_predict.best_ntree_limit, 
                'best_iteration': model_predict.best_iteration, 
                'best_score': model_predict.best_score, 
            }[self.predict_opt]
        elif isinstance(self.predict_opt, int):
            ntree_limit = self.predict_opt
        # prediction
        preds = model_predict.predict(dmat, ntree_limit=ntree_limit)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds

    def predict(self, dmat, fold=None):
        """
            function to make and return prediction
        """
        if fold is None:
            stack_preds = np.zeros((dmat.num_row(), 0))
            for i in range(len(self.models)):
                preds = self._predict_with_option(self.models[i], dmat)
                stack_preds = np.hstack((stack_preds, preds))
        elif isinstance(fold, int):
            stack_preds = self._predict_with_option(self.models[fold], dmat)
        return stack_preds

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

