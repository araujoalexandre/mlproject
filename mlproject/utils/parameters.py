"""
    class parameters for generate 
"""

from os.path import join

class ParametersSpace:

    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        setattr(self, 'data_path', 
               join(self.project_path, 'data'))
        
        for dataset in ['train', 'test']:
            setattr(self, '{}_path'.format(dataset), 
                   join(self.data_path, dataset))
    
    def __repr__(self):
        return str(self.__dict__)