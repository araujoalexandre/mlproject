
from os.path import join
from subprocess import Popen, PIPE, STDOUT
import pandas as pd
import xgboost as xgb

from .base import BaseWrapper
from mlproject.utils import make_directory

# XGB with Python Interface
# XGB with command line

class XGBoostWrapper(BaseWrapper):


    def __init__(self, params):        

        # XXX : check all params
        self.name = 'XGBoost'  

        self.params_booster = params['booster'].copy()
        self.predict_option = params.get('predict_option')
        self._infer_task(params)
        super(XGBoostWrapper, self).__init__(params)

    def _infer_task(self, params):
        # XXX : do this function
        self.task = 'binary'

    def _create_config_file(self, X_train_path, X_cv_path):

        config_path = '{}/config.txt'.format(self.folder)
        with open(config_path, 'r') as f:
            f.write("task = train\n")
            for key, value in self.params_booster.items():
                f.write("{} = {}\n".format(key, value))
            f.write('\n')
            for key, value in self.params:
                f.write("{} = {}\n".format(key, value))
            f.write("data = {}".format(X_train_path))
            f.write("eval[cv] = {}".format(X_cv_path))
            f.write("save_period = 1")
            f.write("model_dir = {}".format(self.folder))

    def train(self, X_train, X_cv, y_train, y_cv, fold):
        """
            Function to train a model
        """
        make_directory(self.folder)
        self.params_booster['evals'] = [(X_train, 'train'), (X_cv, 'cv')]
        self.model = xgb.train(self.params, X_train, **self.params_booster)
        self._features_importance(fold)
        self._dump_txt_model(fold)

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
        fmap_name = join(self.path, "features.map")
        weight = self.model.get_score(fmap=fmap_name, importance_type='weight')
        gain   = self.model.get_score(fmap=fmap_name, importance_type='gain')
        cover  = self.model.get_score(fmap=fmap_name, importance_type='cover')

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
            args_name = [self.folder, self.name, key, fold]
            name = "{}/{}_{}_{}.csv".format(*args_name)
            df.to_csv(name, index=False)


    def _dump_txt_model(self, fold):
        """ 
            make and dump model txt file
        """
        fmap_name = "{}/features.map".format(self.path)
        file_name = "{}/{}_{}.txt".format(self.folder, self.name, fold)
        self.model.dump_model(file_name, fmap=fmap_name, with_stats=True)


    @property
    def get_model(self):
        """
            xxx
        """
        return self.model