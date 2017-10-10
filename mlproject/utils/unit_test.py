
from os.path import join

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from mlproject.utils.serialization import pickle_dump


# class Datasets:

#     def __init__(self):

#         self.nb_features = 10
#         self.seed = 1234

#     def make_binary(self):
        
#         cols = ['v{}'.format(x) for x in range(self.nb_features)]

#         data_params = {
#             'n_samples': 20000,
#             'n_features': self.nb_features,
#             'n_informative': self.nb_features // 2,
#             'random_state': self.seed
#         }

#         X, y = make_classification(**data_params)
#         X_train, X_test = np.array_split(X, 2)
#         y_train, y_test = np.array_split(y, 2)

#         df_train = pd.DataFrame(X_train)
#         df_train.columns = cols
#         df_train['target'] = y_train
#         df_train['id'] = [x for x in range(len(X_train))]

#         df_test = pd.DataFrame(X_test)
#         df_test.columns = cols
#         df_test['target'] = y_test
#         df_test['id'] = [x for x in range(len(X_train))]

#         return df_train, df_test

#     def make_regression(self):
        
#         cols = ['v{}'.format(x) for x in range(self.nb_features)]

#         data_params = {
#             'n_samples': 20000,
#             'n_features': self.nb_features,
#             'n_informative': self.nb_features // 2,
#             'random_state': self.seed
#         }

#         X, y = make_classification(**data_params)
#         X_train, X_test = np.array_split(X, 2)
#         y_train, y_test = np.array_split(y, 2)

#         df_train = pd.DataFrame(X_train)
#         df_train.columns = cols
#         df_train['target'] = df_train[cols].sum(axis=1)
#         df_train['id'] = [x for x in range(len(X_train))]

#         df_test = pd.DataFrame(X_test)
#         df_test.columns = cols
#         df_test['target'] = df_train[cols].sum(axis=1)
#         df_test['id'] = [x for x in range(len(X_train))]

#         return df_train, df_test

#     def make_multiclass(self):
#         pass


def make_binary_dataset(path, nb_features=10, weights=False, groups=False, 
                        seed=10):

        cols = ['v{}'.format(x) for x in range(nb_features)]
        data_params = {
            'n_samples': 20000,
            'n_features': nb_features,
            'n_informative': nb_features // 2,
            'random_state': seed
        }

        X, y = make_classification(**data_params)
        X_train, X_test = np.array_split(X, 2)
        y_train, y_test = np.array_split(y, 2)

        df_train = pd.DataFrame(X_train)
        df_train.columns = cols
        df_train['target'] = y_train
        df_train['id'] = [x for x in range(len(X_train))]

        df_test = pd.DataFrame(X_test)
        df_test.columns = cols
        df_test['target'] = y_test
        df_test['id'] = [x for x in range(len(X_train))]

        # save data
        train_path = join(path, 'data', 'train', 'raw', 'train.csv')
        test_path = join(path, 'data', 'test', 'raw', 'test.csv')
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)

        attributes = ['id', 'target']

        if weights:
            attributes.append('weights')
            pass

        if groups:
            attributes.append('groups')
            pass

        # save attribute data [target / id]        
        for feat in attributes:
            out = join(path, 'data', 'train', 'attributes', 
                            '{}.pkl'.format(feat))
            pickle_dump(df_train[feat].values, out, force=True)
        
            out = join(path, 'data', 'test', 'attributes', 
                            '{}.pkl'.format(feat))
            pickle_dump(df_test[feat].values, out, force=True)

