"""
    class parameters for generate 
"""

from os.path import join

class ParametersSpace:

    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    # XXX : add check for mandatory

    def __repr__(self):
        return str(self.__dict__)

    def as_dict(self):
        return self.__dict__