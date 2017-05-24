"""
__file__

    liblinear.py

__description__

    Liblinear wrapper
    
__author__

    Araujo Alexandre < aaraujo001@gmail.com >

"""

import multiprocessing, copy
import liblinear
import liblinearutil as ll
from .base import Wrapper
from kaggle.utils.functions import make_directory


class Liblinear(Wrapper):


    def __init__(self, params, folder_path):

        self.name = 'Liblinear'

        if params.get('type_solver') in [0,1,2,3,4,5,6,7]:
            self.application = 'classifiction'
        else:
            self.application = 'regression'

        super(Liblinear, self).__init__(params, folder_path)


    def _parse_params(self, params):
        """
            Function to parse parameters
        """
        correspondance = {
            'type_solver': 's',
            'cost' : 'c',
            'epsilon_p': 'p',
            'epsilon_e': 'e',
            'bias': 'B',
            'weight': 'wi',
            'nr_thread': 'n',
            'silent': 'q',
        }

        if params.get('rn_threads') == -1:
            params['rn_threads'] = multiprocessing.cpu_count()

        arr = []
        for key, value in self.params.items():
            if key == 'silent':
                p = '-{}'.format(correspondance[key])
            else:
                p = '-{} {}'.format(correspondance[key], value)
            arr.append(p)

        return ' '.join(arr)


    def train(self, X_train, X_cv, y_train, y_cv):
        """
            Function to train a model
        """
        config = self._parse_params(self.params)
        self.model = ll.train(y_train.tolist(), X_train.tolist(), config)

        make_directory(self.model_folder)


    def predict(self, X, cv=False):
        """
            make prediction
        """
        if self.application == 'classification':
            config = '-b 1 -q'
        else:
            config = '-q'
        y = [0] * len(X)
        if not hasattr(self, 'model'):
            model = ll.load_model(self.out)
            predict = ll.predict(y, X.tolist(), model, config)[0]
        else:
            predict = ll.predict(y, X.tolist(), self.model, config)[0]

        return predict


    @property
    def get_model(self):
        """
            xxx
        """
        args = self.model_folder, self.fold
        self.out = '{}/Liblinear_model_{}.txt'.format(*args)
        ll.save_model(self.out, self.model)
        del self.model
        return copy.copy(self)