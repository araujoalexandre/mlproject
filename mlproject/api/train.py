"""
    train wrapper
"""
import gc
from os.path import join, exists
from datetime import datetime
from copy import deepcopy
from itertools import product
from collections import Iterable

import numpy as np

from mlproject.api.base import BaseAPI
from mlproject.utils.functions import get_ext_cls, counter, gen_zip
from mlproject.utils.serialization import pickle_load, pickle_dump
from mlproject.utils.project import project_path, ProjectPath
from mlproject.utils.log import init_log, print_and_log as print_
from mlproject.utils.log import ProgressTable


# ask to add a comment for the batch of model maybe in the parameters files ?

class TrainWrapper(BaseAPI):

    def __init__(self, path, models_wrapper, **kwargs):

        # path of the model folder
        self.path = path
        self.project = ProjectPath(project_path(self.path))

        # list of all the Wrappers with initilized with parameters
        self.models_wrapper = models_wrapper

        # function to compute the score
        self.metric = kwargs.get('metric')

        # function to make submission file for Data Science Competition
        self.make_submit = kwargs.get('make_submit', None)

        # function for target preprocess and postprocess
        self.target_preprocess = kwargs.get('target_preprocess', lambda x: x)
        self.target_postprocess = kwargs.get('target_postprocess', lambda x: x)

        # print informations and progression table
        self.verbose = True

        # init logger
        if self.verbose:
            self.logger = init_log(self.path)

        # load data
        self._load_data()

    def _startup_message(self):
        print_(self.logger, "\n\nStarting")
        print_(self.logger, "{:%Y.%m.%d %H:%M}\n".format(datetime.now()))
        for model in self.models_wrapper:
            print_(self.logger, model.name)
        print_(self.logger, "")

    def _load_data(self):
        """Load infos.pkl, attributes and validation"""
        infos = dict()
        if exists(join(self.path, 'infos.pkl')):
            infos = pickle_load(join(self.path, 'infos.pkl'))
        self.train_shape = infos.pop('train_shape', None)
        self.test_shape = infos.pop('test_shape', None)
        self.params = infos.pop('project_params', None)
        # load attributes : id, target, groups, weights
        self._load_attributes()
        # load validation indexes
        self.validation = pickle_load(join(self.path, 'validation.pkl'))

    def _load_data_ext(self, path, ext):
        """load a dataset"""
        cls = get_ext_cls()[ext]
        return cls.load(path)

    def split_target(self, tr_ix, va_ix):
        """load and return y splits based on validation index"""
        return self.y_train[tr_ix], self.y_train[va_ix]

    def split_weights(self, tr_ix, va_ix):
        """Load and return weights splited based on validation index"""
        if self.weights_train is not None:
            return self.weights_train[tr_ix], self.weights_train[va_ix]
        return None, None

    def split_groups(self, tr_ix, va_ix):
        """Load and return groups splited based on validation index"""
        if self.groups_train is not None:
            gtr, gva = self.groups_train[tr_ix], self.groups_train[va_ix]
        return None, None

    def load_train(self, fold, seed, ext):
        """Load and return train & cv set from "fold_*" folder"""
        fold_folder = 'fold_{}'.format(fold)
        path_tr = join(self.path, fold_folder, 'X_tr_{}.{}'.format(seed, ext))
        path_va = join(self.path, fold_folder, 'X_va_{}.{}'.format(seed, ext))
        xtr = self._load_data_ext(path_tr, ext)
        xva = self._load_data_ext(path_va, ext)
        return xtr, xva

    def load_test(self, ext):
        """Load the test dataset from "test" folder"""
        path = join(self.path, 'test', 'X_test.{}'.format(ext))
        X_test = self._load_data_ext(path, ext)
        return X_test

    def train_predict(self, save_model=True, submit=True):
        """run models"""

        verbose = self.verbose
        n_samples_train, n_features = self.train_shape
        n_samples_test, _ = self.test_shape
        n_class = self.n_class

        metric = self.metric
        nfolds = self.params.n_folds

        seeds = self.params.seeds
        nbag = len(self.params.seeds)
        
        # print startup message
        if verbose:
            self._startup_message()

        # we iterate over all models define in parameters.py
        for model_id, model_wrapper in enumerate(self.models_wrapper):

            model_id = str(model_id)

            # write header msg
            if verbose:
                print_(self.logger, model_wrapper._print_params())
                # init a new progress table
                progress = ProgressTable(self.logger)

            # load test set
            xtest = self.load_test(model_wrapper.ext)

            # define model name and model folder to model cls
            model_wrapper.name = '_'.join([model_wrapper.name, model_id])
            model_wrapper.folder = '_'.join([model_wrapper.folder, model_id])

            # duplicate model if bagging activated : multiple seeds
            bag_of_model = [deepcopy(model_wrapper) for _ in range(nbag)]

            # out of fold prediction for stacking
            train_stack = np.zeros((n_samples_train, n_class))
            test_stack = np.zeros((n_samples_test, n_class*nfolds))

            # bagged scores
            all_scores_tr, all_scores_va = [], []

            # loop over duplicated bagged models
            for _, (model, idx, seed) in gen_zip(bag_of_model, self.validation, 
                                                    seeds):

                # transfer seed to model
                model.seed = seed

                # model scores
                scores_tr, scores_va = [], []

                # timer
                start_loop = datetime.now()
                # start training loop
                for fold, (tr_ix, va_ix) in enumerate(idx):

                    # load y, weights, train
                    ytr, yva = self.split_target(tr_ix, va_ix)
                    wtr, wva = self.split_weights(tr_ix, va_ix)
                    gtr, gva = self.split_groups(tr_ix, va_ix)
                    xtr, xva = self.load_train(fold, seed, model.ext)

                    # target preprocess : transform the target
                    ytr = self.target_preprocess(ytr)
                    yva = self.target_preprocess(yva)

                    # train
                    start = datetime.now()
                    model.train(xtr, xva, ytr, yva, fold)
                    end = datetime.now()

                    # make prediction
                    ytr_hat = model.predict(xtr, fold=fold)
                    yva_hat = model.predict(xva, fold=fold)

                    # target postprocess : reverse the target transform
                    ytr, yva = self.split_target(tr_ix, va_ix)
                    ytr_hat = self.target_postprocess(ytr_hat)
                    yva_hat = self.target_postprocess(yva_hat)

                    # score model
                    tr_error = metric(ytr, ytr_hat, weights=wtr, groups=gtr)
                    va_error = metric(yva, yva_hat, weights=wva, groups=gva)
                    scores_tr += [tr_error]
                    scores_va += [va_error]
                    all_scores_tr += [tr_error]
                    all_scores_va += [va_error]

                    # accumulate predictions
                    train_stack[va_ix, :] += yva_hat

                    # update progress table
                    progress.score(seed, fold, tr_error, va_error, start, end)

                    del ytr, yva, wtr, wva, gtr, gva, xtr, xva
                    gc.collect()
                
                # timer
                end_loop = datetime.now()

                mean_tr, mean_va = np.mean(scores_tr), np.mean(scores_va)
                progress.score('/', '/', mean_tr, mean_va, start_loop, end_loop)

                # predict each fold model on test dataset
                test_stack += self.target_postprocess(model.predict(xtest))

            # averaging probabilities by the number of model in bagging
            train_stack /= nbag
            test_stack /= nbag
            test_stack =  test_stack.mean(1)

            # compute total score
            score = metric(self.y_train, train_stack, 
                           weights=self.weights_train, 
                           groups=self.groups_train
                        )

            if verbose:
                stats = [np.std(all_scores_va), np.var(all_scores_va)]
                diff = score - np.mean(all_scores_va)
                # print infos
                msg = (
                    "score on average train oof : {:.5f}\n"
                    "validation stats : std : {:.5f}, var : {:.5}\n"
                    "diff (train_oof - mean valid) : {:.5f}\n"
                    )
                print_(self.logger, msg.format(score, *stats, diff))

            pickle_dump(train_stack, join(model.folder, 'train_stack.pkl'))
            pickle_dump(test_stack, join(model.folder, 'test_stack.pkl'))

            # make submit if submit is True and id self.id_test exist
            if submit and hasattr(self, 'id_test'):
                # compute test score if we have the test target
                score_test = None
                if self.y_test is not None:
                    score_test = metric(self.y_test, test_stack,
                                        weights=self.weights_test,
                                        groups=self.groups_test)
                # make submission file
                score_test = score_test or 0.0
                self.make_submit(model.folder, self.id_test, test_stack,
                        model.name, model.date, score, score_test)

            if save_model:
                model.save_model()


