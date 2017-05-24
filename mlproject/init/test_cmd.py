

import os, sys

if not os.path.exists('code'):
    os.makedirs('code')
if not os.path.exists('jupyter'):
    os.makedirs('jupyter')
if not os.path.exists('models'):
    os.makedirs('models')
for folder_type in ['train', 'test']:
    if not os.path.exists('data/{}/pkl'.format(folder_type)):
        os.makedirs('data/{}/pkl'.format(folder_type))
    if not os.path.exists('data/{}/original'.format(folder_type)):
        os.makedirs('data/{}/original'.format(folder_type))
    if not os.path.exists('data/{}/features'.format(folder_type)):
        os.makedirs('data/{}/features'.format(folder_type))