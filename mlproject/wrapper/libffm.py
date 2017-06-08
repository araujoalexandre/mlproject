
import sys, os, copy
import multiprocessing
from subprocess import Popen, PIPE, STDOUT
import datetime
import numpy as np
import pandas as pd

from .base import BaseWrapper
from mlproject.utils import make_directory


class LibFFMWrapper(BaseWrapper):

    def __init__(self, params, paths):

        self.name = 'LibFFM'
        self.file_ext = 'ffm'

        self.application = params.get('application')
        self.exec_train = '/home/alexandrearaujo/library/libffm/ffm-train'
        self.exec_predict = '/home/alexandrearaujo/library/libffm/ffm-predict'
        # self.predict_increment = 0
        super(LibFFMWrapper, self).__init__(params, paths)

    def _parse_params(self, params):
        """
            Function to parse parameters
        """
        correspondance = {
            'lambda': 'l',
            'factor': 'k',
            'iteration': 't',
            'eta': 'r',
            'nr_threads': 's'
        }
        params['nr_threads'] = multiprocessing.cpu_count() - 5

        arr = []
        for key, value in self.params.items():
            if key == 'iteration' and isinstance(value, list):
                arr.append('-{}'.format(correspondance[key]))
                arr.append('{}'.format(value[self.fold]))
                continue
            arr.append('-{}'.format(correspondance[key]))
            arr.append('{}'.format(value))

        return arr


    def train(self, X_train, X_cv, y_train, y_cv):
        """
            Function to train a model
        """

        make_directory(self.model_folder)

        args = [self.folder_path, self.fold]
        # train_path = "{}/Fold_{}/X_train.libffm".format(*args)
        # cv_path = "{}/Fold_{}/X_cv.libffm".format(*args)

        train_path = X_train
        cv_path = X_cv

        out = '{}/LibFFM_model_{}.txt'.format(self.model_folder, self.fold)

        cmd = []
        cmd.append(self.exec_train)
        cmd.extend(['-p', cv_path])
        cmd.extend(self._parse_params(self.params))
        cmd.append('--auto-stop')
        # cmd.append('--no-norm')
        # cmd.append('--on-disk')

        # CUSTOM LIB
        # cmd.append('--group')
        # cmd.append('{}/Fold_{}/'.format(self.folder_path, self.fold))
        # cmd.append('--test')
        # cmd.append('{}/Test/X_test.libffm'.format(self.folder_path))
        # cmd.append('--folder_path')
        # cmd.append('{}'.format(self.model_folder))
        # cmd.append('--fold')
        # cmd.append('{}'.format(self.fold))

        cmd.append(train_path)
        cmd.append(out)

        print(cmd, flush=True)

        # process = Popen(cmd, shell=True, stdout=PIPE, bufsize=1).communicate()

        with Popen(cmd, stdout=PIPE, bufsize=1, universal_newlines=True) as p:
            for line in p.stdout:
                line = line.strip()
                print(line, flush=True)



    def predict(self, X, path=None, cv=False):
        """
            function to make and return prediction
        """

        if self.dataset == 'train':
            args = [self.folder_path, self.fold, self.ext]
            if self.predict_increment == 0:
                predict_filepath = "{}/Fold_{}/X_train.{}".format(*args)
                self.predict_increment += 1
            elif self.predict_increment == 1:
                predict_filepath = "{}/Fold_{}/X_cv.{}".format(*args)
                self.predict_increment = 0
        elif self.dataset == 'test':
            args = [self.folder_path, self.ext]
            predict_filepath = "{}/Test/X_test.{}".format(*args)

        if path is not None:
            predict_filepath = path

        args = [self.model_folder, self.fold]
        input_model = "{}/LibFFM_model_{}.txt".format(*args)
        output_results = "{}/LibFFM_preds_{}.txt".format(*args)

        cmd = []
        cmd.append(self.exec_predict)
        cmd.append(predict_filepath)
        cmd.append(input_model)
        cmd.append(output_results)

        process = Popen(cmd, stdout=PIPE, bufsize=1)
        process.wait()


        # # custom
        # if self.dataset == 'train':
        #     if cv == False:
        #         output_results = "{}/LibFFM_preds_{}_{}.txt".format(self.model_folder, self.dataset, self.fold)
        #     elif cv == True:
        #         output_results = "{}/LibFFM_preds_{}_{}.txt".format(self.model_folder, 'cv', self.fold)
        # else:
        #     output_results = "{}/LibFFM_preds_{}_{}.txt".format(self.model_folder, self.dataset, self.fold)

        prections = pd.read_csv(output_results, header=None)[0].values
        return prections


    def _dump_txt_model(self, fold):
        """ 
            make and dump model txt file
        """
        fmap_name = "{}/features.map".format(self.folder_path)
        file_name = "{}/{}_{}.txt".format(self.model_folder, self.name, fold)
        self.model.dump_model(file_name, fmap=fmap_name, with_stats=True)


    @property
    def get_model(self):
        """
            xxx
        """
        model = copy.copy(self)
        model.dataset = 'test'
        return model