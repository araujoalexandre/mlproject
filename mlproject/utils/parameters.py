"""
    class parameters for generate 
"""

from os.path import join

class ParametersSpace:

    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        attributes = (
            'project_path', 'data_path', 
            'train_name', 'test_name'
            'id_train', 'id_test',
            'target_train', 'target_test',
            'weights_train', 'weights_test',
            'groups_train', 'groups_test',
            'n_folds', 'seed','missing',
        )
        # set default attr if attr has not been set
        for attr in attributes:
            if not hasattr(self, attr):
                setattr(self, key, None)

    def __repr__(self):
        return str(self.__dict__)

    def as_dict(self):
        return self.__dict__