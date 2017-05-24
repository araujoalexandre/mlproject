
"""
__file__

    functions.py

__description__

    This file provides various functions.
    
__author__

    Araujo Alexandre < alexandre.araujo@wavestone.fr >

"""

import os, sys

def startprojet():
    """
        Create and init folder projet
    """
    path = os.getcwd()
    folder = input("Folder Name : ")
    if folder != '':
        if not os.path.exists('{}/{}'.format(path, folder)):
            os.makedirs('{}/{}'.format(path, folder))
            path += folder
        else:
            # use Raise instead
            sys.exit("Folder already exists")
    print(path)

    if not os.path.exists('{}/code'.format(path)):
        os.makedirs('{}/code'.format(path))
    if not os.path.exists('{}/jupyter'.format(path)):
        os.makedirs('{}/jupyter'.format(path))
    if not os.path.exists('{}/models'.format(path)):
        os.makedirs('{}/models'.format(path))
    for folder_type in ['train', 'test']:
        if not os.path.exists('{}/data/{}/pkl'.format(path, folder_type)):
            os.makedirs('{}/data/{}/pkl'.format(path, folder_type))
        if not os.path.exists('data/{}/original'.format(path, folder_type)):
            os.makedirs('{}/data/{}/original'.format(path, folder_type))
        if not os.path.exists('data/{}/features'.format(path, folder_type)):
            os.makedirs('{}/data/{}/features'.format(path, folder_type))

if __name__ == "__main__":
    startprojet()