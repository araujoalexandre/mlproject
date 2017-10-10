"""
__file__

    functions.py

__description__

    This file provides pprint class.
    
__author__

    Araujo Alexandre < alexandre.araujo@wavestone.fr >

"""

import sys
from os.path import join
from inspect import isfunction


def init_log(path):
    f = open(join(path, 'logs.log'), 'a')
    return f

def print_and_log(logger, string):
    """print and log string"""
    print(string, flush=True)
    logger.write("{}\n".format(str(string)))
    logger.flush()


def pprint_dict(data, tab=0, out=''):
    """recusive walk in dict, output dict as string"""
    for key in sorted(data.keys()):
        if isinstance(data[key], dict):
            out = pprint_dict(data[key], tab=tab+1, out=out)
            continue
        strtab = '  '*tab
        if isfunction(data[key]):
            data[key] = data[key].__name__
        out += '{}{}: {}\n'.format(strtab, key, data[key])
    return out[:-1]


class ProgressTable:

    def __init__(self, logger=None, padding=13):

        self.headers = ['SEED','FOLD','TRAIN','CV','START','END','DUR']
        self.len_title = len(self.headers)
        self.padding = padding

        self.logger = logger

        self.initialize = True

    def _format_timedelta(self, timedeltaObj):
        """Convert timedelta Object to date string"""
        s = timedeltaObj.total_seconds()
        arr = [int(s // 3600), int(s % 3600 // 60), int(s % 60)]
        return '{:02d}:{:02d}:{:02d}'.format(*arr)

    def _title(self):
        """print table headers"""
        arr = ["|{x: ^{fill}}".format(x=x, fill=self.padding).format(x) 
                                                          for x in self.headers]
        string = "".join(arr) + "|"
        print_and_log(self.logger, self._line)
        print_and_log(self.logger, string)
        print_and_log(self.logger, self._line)

    def score(self, seed, fold, train, cv, start, end):
        """print score and other info """
        if self.initialize: 
            self._title()
            self.initialize = False
        if not isinstance(fold, int):
            print_and_log(self.logger, self._line)

        dur = self._format_timedelta(end - start)
        arr = [ seed, fold, '{:.5f}'.format(train), '{:.5f}'.format(cv), 
                start.strftime("%H:%M:%S"), end.strftime("%H:%M:%S"), dur]
        arr = ["|{x: ^{fill}}".format(x=x, fill=self.padding).format(x) 
                                                                for x in arr]
        string = "".join(arr) + "|"
        print_and_log(self.logger, string)
        if not isinstance(fold, int): print_and_log(self.logger, self._line)

    @property
    def _line(self):
        """print line of score table in training"""
        return "-" * (self.padding * self.len_title + self.len_title + 1)