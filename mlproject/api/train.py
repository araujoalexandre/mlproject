"""
    train wrapper
"""
import gc
from os.path import join, exists
from datetime import datetime
from copy import deepcopy
from itertools import product

import numpy as np

from mlproject.api import BaseAPI, TransformAPI
from mlproject.utils import get_ext_cls
from mlproject.utils import pickle_load, pickle_dump
from mlproject.utils import init_log
from mlproject.utils import project_path, ProjectPath
from mlproject.utils import print_and_log as print_
from mlproject.utils import ProgressTable
from mlproject.utils import counter


# ask to add a comment for the batch of model maybe in the parameters files ?
# save train_stack (rename it) as his own file
# option to save the model (default True)
# save features importance in his own folder


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

        # function for target procession
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
        """
        """
        print_(self.logger, "\n\nStarting")
        print_(self.logger, "{:%Y.%m.%d %H:%M}\n".format(datetime.now()))
        for model in self.models_wrapper:
            print_(self.logger, model.name)
        print_(self.logger, "")

    def _load_data(self):
        """
            function to load important data files :
                id [train/test]
                y  [train/test]
                weights [train/test]
                groups [train/test]
        """
        infos = dict()
        if exists(join(self.path, "infos.pkl")):
            infos = pickle_load(join(self.path, "infos.pkl"))
        self.train_shape = infos.pop("train_shape", None)
        self.test_shape = infos.pop("test_shape", None)
        self.params = infos.pop("project_params", None)

        self._load_id()
        self._load_target()
        self._load_weights()
        self._load_groups()

        self.validation = pickle_load(join(self.path, "validation.pkl"))

    def _load_data_ext(self, path, ext):
        """
            private function to load a dataset
        """
        cls = get_ext_cls()[ext]
        data = cls.load(path)
        return data

    def split_target(self, tr_ix, va_ix):
        """
            Load and return y splits based on validation index
        """
        return self.y_train[tr_ix], self.y_train[va_ix]

    def split_weights(self, tr_ix, va_ix):
        """
            if weights exists, Load and return weights
            splits based on validation index
        """
        if self.weights_train is not None:
            return self.weights_train[tr_ix], self.weights_train[va_ix]
        return None, None

    def split_groups(self, tr_ix, va_ix):
        """
            if weights exists, Load and return weights
            splits based on validation index
        """
        if self.groups_train is not None:
            gtr, gva = self.groups_train[tr_ix], self.groups_train[va_ix]
        return None, None

    def load_train(self, fold, seed, ext):
        """
            Load and return train & cv set from "fold_*" folder
        """
        fold_folder = "fold_{}".format(fold)
        path_tr = join(self.path, fold_folder, "X_tr_{}.{}".format(seed, ext))
        path_va = join(self.path, fold_folder, "X_va_{}.{}".format(seed, ext))
        xtr = self._load_data_ext(path_tr, ext)
        xva = self._load_data_ext(path_va, ext)
        return xtr, xva

    def load_test(self, ext):
        """
            Load the test dataset from "test" folder
        """
        path = join(self.path, "test", "X_test.{}".format(ext))
        X_test = self._load_data_ext(path, ext)
        return X_test

    def models_loop(self, save_model=True):
        """
            training loop
        """
        verbose = self.verbose
        train_shape = self.train_shape
        metric = self.metric
        nfolds = len(self.validation)
        nseed = len(self.params.seed)

        # container for fitted models
        self.fitted_models = []

        # print startup message
        if verbose:
            self._startup_message()

        # iterator for future value
        iter_validation = iter(self.validation)
        # id to flag new bagging iteration
        prev_model_id = None
        self.models_id = [x for x in range(len(self.models_wrapper))]

        loop = product(
                    enumerate(self.models_wrapper),
                    enumerate(self.params.seed)
                    )
        for enum, ((model_id, model), (seed_id, seed)) in enumerate(loop):

            # save wrapper for test prediction
            self.fitted_models += [deepcopy(model)]
            model = self.fitted_models[-1]

            # transfer seed to model
            model.seed = seed

            # extract extension to use
            ext = model.ext

            # (re-)init a new bagged model
            if prev_model_id != model_id:

                # add enum to model.name and model.folder
                model_name = "{}_{}".format(model.name, model_id)
                model_folder = "{}_{}".format(model.folder, model_id)

                # reset
                prev_model_id = model_id
                # out of fold prediction for stacking
                train_stack = None
                # bagged scores
                all_scores_tr, all_scores_va = [], []
                # write header msg
                if verbose:
                    print_(self.logger, model._print_params())
                    # print_(self.logger, '')
                    # print_(self.logger, model_name)
                    # print_(self.logger, model)  # print model params
                    # init a new progress table
                    progress = ProgressTable(self.logger)

            model.name = model_name
            model.folder = model_folder

            # model scores
            scores_tr, scores_va = [], []

            # timer
            start_loop = datetime.now()
            # start training loop
            for fold, (tr_ix, va_ix) in enumerate(
                                        self.validation[enum % nseed]):

                # load y, weights, train
                ytr, yva = self.split_target(tr_ix, va_ix)
                wtr, wva = self.split_weights(tr_ix, va_ix)
                gtr, gva = self.split_groups(tr_ix, va_ix)
                xtr, xva = self.load_train(fold, seed, ext)

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

                # fill train stacking dataset
                if train_stack is None:
                    train_stack = np.zeros((train_shape[0], yva_hat.shape[1]))
                # accumulate probabilities
                train_stack[va_ix, :] += yva_hat

                # update progress table
                progress.score(seed, fold, tr_error, va_error, start, end)

                del ytr, yva, wtr, wva, gtr, gva, xtr, xva
                gc.collect()

            # timer
            end_loop = datetime.now()

            mean_tr = np.mean(scores_tr)
            mean_va = np.mean(scores_va)
            progress.score('/', '/', mean_tr, mean_va, start_loop, end_loop)

            # compute total score
            model.score = metric(
                            self.y_train,
                            (train_stack / (seed_id+1)),
                            weights=self.weights_train,
                            groups=self.groups_train
                            )

            # end of bagged model
            if enum % nseed == (nseed - 1):

                # averaging probabilities
                train_stack /= nseed

                if verbose:
                    stats = [np.std(all_scores_va), np.var(all_scores_va)]
                    diff = model.score - np.mean(all_scores_va)
                    # print infos
                    msg = (
                        "score on average train oof : {:.5f}\n"
                        "validation stats : std : {:.5f}, var : {:.5}\n"
                        "diff (train_oof - mean valid) : {:.5f}\n"
                        )
                    print_(self.logger, msg.format(model.score, *stats, diff))

                pickle_dump(train_stack, join(model_folder, "train_stack.pkl"))

            if save_model:
                model.save_model()

    def predict_test(self, compute_score=False, submit=True):
        """
            loop over fitted model for prediction
        """
        test_shape = self.test_shape
        metric = self.metric
        nfolds = len(self.validation[0])
        nseed = len(self.params.seed)

        for enum, model in enumerate(self.fitted_models):

            # extract file extension to use
            ext = model.ext

            # load test set
            xtest = self.load_test(ext)

            # prediction on test set from all fold model
            fold_preds = model.predict(xtest)

            # target postprocess : reverse the target transform
            fold_preds = self.target_postprocess(fold_preds)

            nclass = fold_preds.shape[1] // nfolds

            if nclass > 1:
                test_stack = np.zeros((test_shape[0], nclass))
                for x in range(nclass):
                    test_stack[:, x] = fold_preds[:, x::nclass].mean(axis=1)
            else:
                test_stack = fold_preds.mean(axis=1)

            if enum % nseed == (nseed - 1):

                pickle_dump(test_stack, join(model.folder, "test_stack.pkl"))

                if submit and hasattr(self, 'id_test'):

                    score_test = None
                    if self.y_test is not None:
                        score_test = metric(self.y_test, test_stack,
                                            weights=self.weights_train,
                                            groups=self.groups_train)

                    score_test = score_test or 0.0
                    args = []
                    self.make_submit(model.folder, self.id_test, test_stack,
                            model.name, model.date, model.score, score_test)
