"""
__file__

    functions.py

__description__

    This file provides pprint class.
    
__author__

    Araujo Alexandre < alexandre.araujo@wavestone.fr >

"""

class pprint:

    def __init__(self, logger=None, padding=14):

        self.logger = logger
        self.padding = padding


    def to_print(self, string):
        """
            xxx
        """
        print(string)
        if self.logger != None:
            self.logger.info(string)


    def _format_timedelta(self, timedeltaObj):
        """
            Convert timedelta Object to date string
        """
        s = timedeltaObj.total_seconds()
        arr = [int(s // 3600), int(s % 3600 // 60), int(s % 60)]
        return '{:02d}:{:02d}:{:02d}'.format(*arr)


    def title(self, titles=['FOLD','TRAIN','CV','START','END','DUR']):
        """
            xxx
        """
        self.len_title = len(titles)
        line = "-" * (self.padding * len(titles) + len(titles) + 1)
        arr = ["|{x: ^{fill}}".format(x=x, fill=self.padding).format(x) 
                                                                for x in titles]
        string = ''.join(arr) + "|"
        self.to_print(line)
        self.to_print(string)
        self.to_print(line)


    def score(self, fold, train, cv, start, end):
        """
            xxx
        """
        dur = self._format_timedelta(end - start)
        arr = [ fold, '{:.5f}'.format(train), '{:.5f}'.format(cv), 
                start.strftime("%H:%M:%S"), end.strftime("%H:%M:%S"), dur]
        padding = self.padding
        arr = ["|{x: ^{fill}}".format(x=x, fill=padding).format(x) for x in arr]
        string = ''.join(arr) + "|"
        self.to_print(string)


    def line(self):
        """
            xxx
        """
        line = "-" * (self.padding * self.len_title + self.len_title + 1)
        self.to_print(line)