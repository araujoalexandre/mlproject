from os import getcwd
from os.path import exists, join
from datetime import datetime
from contextlib import redirect_stdout

import numpy as np

class BaseWrapper:

    def __init__(self, params):

        self.path = getcwd()
        self.date = datetime.now()

        self.name = "{}_{:%m%d%H%M}".format(self.name, self.date)
        self.folder = join(self.path, self.name)

        # XXX : check all params
        self.params = params['params'].copy()
        self.ext    = params.pop('ext')
        
        self.task = None

        self.y = None
        self.groups = None
        self.weights = None

    # def __str__(self):
    #     if len(self.params.items()) > 0:
    #         return str(self.params)
    #     return ''

    def _print_params(self):
        """ print params for logs """
        # XXX : maybe convert this function as string for formating
        message = "\n\n{}\n".format(self.name)
        if hasattr(self, "booster"):
            message += "{}\n".format(str(self.booster))
        message += "{}".format(str(self.params))
        return message

    def train(self, *args, **kwargs):
        with open(join(self.path, 'verbose_model.log'), 'a') as f:
            with redirect_stdout(f):
                print(self._print_params())
                self._train(*args, **kwargs)