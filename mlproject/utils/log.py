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
from logging import getLogger, basicConfig, INFO


def init_log(path):
    filename = join(path, 'logs.log')
    logger = getLogger()
    basicConfig(filename=filename, level=INFO)
    return logger

def print_and_log(logger, string):
    """
        print and log string
    """
    print(string)
    if logger != None:
        logger.info(string)

# def progress_bar(enum):
#     bar_size = 40
#     sys.stdout.write('[{}]'.format(' '*bar_size))
#     sys.stdout.flush()
#     sys.stdout.write("\b"*(bar_size+1))
#     dot = len(enum) // 40
#     for i, x in enumerate(enum):
#         if i % dot == 0:
#             sys.stdout.write("-")
#             sys.stdout.flush()
#         yield x


class ProgressTable:

    def __init__(self, logger=None, padding=14):

        self.headers = ['FOLD', 'TRAIN', 'CV', 'START', 'END', 'DUR']
        self.len_title = len(self.headers)
        self.padding = padding

        # XXX : add condition if logger == None
        self.logger = logger

    def _format_timedelta(self, timedeltaObj):
        """
            Convert timedelta Object to date string
        """
        s = timedeltaObj.total_seconds()
        arr = [int(s // 3600), int(s % 3600 // 60), int(s % 60)]
        return '{:02d}:{:02d}:{:02d}'.format(*arr)

    def _title(self):
        """
            print table headers
        """
        arr = ["|{x: ^{fill}}".format(x=x, fill=self.padding).format(x) 
                                                          for x in self.headers]
        string = ''.join(arr) + "|"
        self.to_print('{}\n{}\n{}'.format(self._line(), string, self._line()))

    def score(self, fold, train, cv, start, end):
        """
            print score and other info 
        """
        if fold == 0: self._title()
        if fold == None: self.to_print(self._line())

        dur = self._format_timedelta(end - start)
        arr = [ fold, '{:.5f}'.format(train), '{:.5f}'.format(cv), 
                start.strftime("%H:%M:%S"), end.strftime("%H:%M:%S"), dur]
        arr = ["|{x: ^{fill}}".format(x=x, fill=self.padding).format(x) 
                                                                for x in arr]
        string = ''.join(arr) + "|"
        print_and_log(self.logger, string)
        
        if fold == None: print_and_log(self.logger, self._line())

    def _line(self):
        """
            print line of score table in training
        """
        return "-" * (self.padding * self.len_title + self.len_title + 1)