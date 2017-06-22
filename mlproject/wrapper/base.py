from os import getcwd
from os.path import exists, join
from datetime import datetime
from contextlib import redirect_stdout

import numpy as np

class BaseWrapper:

    def __init__(self, params):

        self.path = getcwd()
        self.date = datetime.now()

        name = "{}_{:%m%d%H%M}".format(self.name, self.date)
        self.folder = join(self.path, name)

        # XXX : check all params
        self.params = params.pop('params')
        self.ext = params.pop('ext')
        self.n_jobs = params.get('n_jobs', -1)

        self.task = None

        self.y = None
        self.groups = None
        self.weights = None

    def __str__(self):
        if len(self.params.items()) > 0:
            return str(self.params)
        return ''

    def train(self, *args, **kwargs):
        with open(join(self.path, 'verbose_model.log'), 'a') as f:
            with redirect_stdout(f):
                print("\n\n{}".format(self.name))
                print(self.params)
                self._train(*args, **kwargs)