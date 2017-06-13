"""
    train wrapper
"""
from os.path import join, exists
from datetime import datetime

import numpy as np

from mlproject.utils import pickle_load, pickle_dump
from mlproject.utils import init_log
from mlproject.utils import print_and_log as print_
from mlproject.utils import ProgressTable


# ask to add a comment for the batch of model maybe in the parameters files ?
# save train_stack (rename it) as his own file
# option to save the model (default True)
# save features importance in his own folder


class TrainWrapper:

    def __init__(self, path, models_wrapper, **kwargs):

        # path of the model folder
        self.path = path
        
        # list of all the Wrappers with initilized with parameters
        self.models_wrapper = models_wrapper
        
        # function to compute the score
        self.metric = kwargs.get('metric')
        
        # function to make submission file for Data Science Competi
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

    def _load_data(self):
        """
            function to load important data files : 
                id [train/test]
                y  [train/test]
                weights [train/test]
                groups [train/test]
        """
        self.infos = pickle_load(join(self.path, "infos.pkl"))
        self.train_shape = self.infos.get("train_shape")
        self.test_shape = self.infos.get("test_shape")

        for dataset in ["train", "test"]:
            if dataset == "test" and self.test_shape is (None, None): 
                continue
            for file in ["id", "y", "weights", "groups"]:
                filename = '_'.join([file, dataset])
                file_path = join(self.path, filename)
                if exists(file_path):
                    setattr(self, filename, pickle_load(file_path))
        self.validation = pickle_load(join(self.path, "validation.pkl"))

    def _startup_message(self):
        """
        """
        print_(self.logger, "\n\nStarting")
        print_(self.logger, "{:%Y.%m.%d %H:%M}\n".format(datetime.now()))
        for model in self.models_wrapper:
            print_(self.logger, model.name)
        print_(self.logger, "")

    def models_loop(self, save_model=True):
        """
            training loop 
        """
        verbose = self.verbose
        train_shape = self.train_shape
        validation = self.validation
        nfolds = len(validation)

        # print startup message 
        if verbose:
            self._startup_message()

        for enum, model in enumerate(self.models_wrapper):

            num_class = 1
            # override num_class
            if model.task == 'multiclass':
                num_class = len(set(self.y_train))

            model.name    = "{}_{}".format(model.name, enum)
            model.folder  = "{}_{}".format(model.folder, enum)
            
            for dataset in ["train", "test"]:
                if dataset == "test" and self.test_shape is (None, None): 
                    continue
                for attr_type in ["y", "weights", "group"]:
                    attr = '_'.join([attr_type, dataset])
                    if hasattr(self, attr):
                        setattr(model, attr, getattr(self, attr))

            # stacking
            train_stack = np.zeros((train_shape[0], num_class))

            scores_tr, scores_cv = [], []
            self.fitted_models = []

            if verbose:
                print_(self.logger, '')
                print_(self.logger, model.name)
                print_(self.logger, model) # print model params
                progress = ProgressTable(self.logger)

            # timer
            start_loop = datetime.now()
            # start training loop
            for fold, (tr_ix, va_ix) in enumerate(validation):
                
                # load y, weights, train
                ytr, ycv = model.split_target(tr_ix, va_ix)
                wtr, wcv = model.split_weights(tr_ix, va_ix)
                gtr, gcv = model.split_groups(tr_ix, va_ix)
                xtr, xcv = model.split_train(fold)
                
                # train
                start = datetime.now()
                model.train(xtr, xcv, ytr, ycv, fold)
                end = datetime.now()
                
                # make prediction
                ytr_hat = model.predict(xtr)
                ycv_hat = model.predict(xcv)

                # process dim y_hat
                if ytr_hat.ndim == 1:
                    ytr_hat = ytr_hat.reshape(-1, 1)
                if ycv_hat.ndim == 1:
                    ycv_hat = ycv_hat.reshape(-1, 1)

                # evaluating model => XXX load metric
                tr_error = self.metric(ytr, ytr_hat, weights=wtr, groups=gtr)
                cv_error = self.metric(ycv, ycv_hat, weights=wcv, groups=gcv)
                scores_tr += [tr_error]
                scores_cv += [cv_error]

                # filling train stacking dataset
                train_stack[va_ix, :] = ycv_hat

                # update progress table
                progress.score(fold, tr_error, cv_error, start, end)

            # timer
            end_loop = datetime.now()

            # compute total score
            model.score = self.metric(  model.load_target(), 
                                        train_stack, 
                                        weights=model.load_weights(), 
                                        groups=model.load_groups())
            mean_tr = np.mean(scores_tr)
            mean_cv = np.mean(scores_cv)
            stats = [np.std(scores_cv), np.var(scores_cv)] 
            diff = model.score - np.mean(scores_cv)

            if verbose:
                # print infos
                progress.score('', mean_tr, mean_cv, start_loop, end_loop)

                print_(self.logger, "score train oof : {:.5f}".format(model.score))
                print_(self.logger, ("validation stats :  Std : {:.5f}, "
                                 "Var : {:.5f}").format(*stats))
                print_(self.logger, "Diff FULL TRAIN  - MEAN CV: {:.5f}".format(diff))
                print_(self.logger, '')

            pickle_dump(train_stack, join(model.folder, "train_stack.pkl"))
            # if save_model:
            #     # pickle dump to right folder
            #     model.get_model

    def predict_test(self, compute_score=False, submit=True):
        """
        """
        test_shape = self.test_shape
        nfolds = len(self.fitted_models)
        for enum, model in enumerate(self.models_wrapper):
            
            # load test set
            xtest = model.load_test()

            # num_class is 1 for binary classifaction and regression
            num_class = 1

            # check task 
            multiclass = True if model.task == 'multiclass' else False
            if multiclass:
                num_class = len(set(self.y_train))

            # init test_stack
            test_stack = np.zeros((test_shape[0], nfolds*num_class))

            for fold in range(nfolds):

                # evaluating model
                y_hat = model.predict(xtest)

                # process dim y_hat
                if y_hat.ndim == 1:
                    y_hat = y_hat.reshape(-1, 1)

                # filling test stacking dataset
                if multiclass:
                    index = np.array(list(range(num_class)))
                    test_stack[:, index + fold*num_class] = y_hat
                else:
                    test_stack[:, fold] = y_hat

            if multiclass:
                prediction = np.zeros((test_shape[0], num_class))
                for x in range(num_class):
                    prediction[:, x] = test_stack[:, x::num_class].mean(axis=1)
            else:
                prediction = test_stack.mean(axis=1)
            pickle_dump(test_stack, join(model.folder, "test_stack.pkl"))

            if submit:
                self.make_submit(self.id_test, prediction, model.path, 
                    model.name, model.date, model.score)