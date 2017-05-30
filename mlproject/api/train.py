"""
__file__

    train.py

__description__

    This file trains the models find in parameters file.
    
__author__

    Araujo Alexandre < alexandre.araujo@wavestone.fr >

"""

from logging import getLogger, basicConfig, INFO
from datetime import datetime
import os, sys, glob, pickle

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss, roc_auc_score

from kaggle.utils.functions import pickle_dump, pickle_load
from kaggle.utils.functions import CustomPrint, make_submit
from kaggle.utils.functions import target_transform
from parameters import get_params


def metric(y_true, y_hat, groups=None, weight=None):
    """
        custom metric for evaluation
    """
    return log_loss(y_true, y_hat, sample_weight=weight)


def custom_submit(id_test, preds, *args, **kwargs):
    """
        Function to create sumbit file for Kaggle competition
    """
    df_submit = pd.DataFrame({'test_id':id_test, 'is_duplicate': preds})
    file_name = "{}/submit_{}_{}_{:.5f}_0.00000.csv.gz".format(*args)
    df_submit.to_csv(file_name, index=False, compression='gzip')


# ask to add a comment for the batch of model maybe in the parameters files ?
# save train_stack (rename it) as his own file
# option to save the model (default True)
# save features importance in his own folder


# class TrainWrapper:
#     XXX

    

def main():
    
    ######################
    #####    INIT    #####
    ######################
    
    models_to_run = get_params()
    do_submit = True

    logger = init_log()
    print_ = CustomPrint(logger)

    print_.to_print('')
    print_.to_print('')
    print_.to_print("Starting")
    print_.to_print(datetime.now().strftime("%Y.%m.%d %H:%M"))
    print_.to_print('')

    for model in models_to_run:
        print_.to_print(model.name)
    print_.to_print('')

    #######################
    #####    INFOS    #####
    #######################

    folder = os.path.split(os.path.realpath(__file__))[0]
    infos = pickle_load('{}/infos.pkl'.format(folder))
    
    n_folds = infos.get('n_folds')
    train_shape = infos.get('train_shape')
    test_shape = infos.get('test_shape')
    folder_path = infos.get('folder_path')
    data_path = infos.get('data_path')

    ######################
    #####    LOAD    #####
    ######################
    
    if test_shape is not None:
        id_test = pickle_load("{}/id_test.pkl".format(folder_path))
    y_true = pickle_load("{}/y_true.pkl".format(folder_path))

    if os.path.exists("{}/weights.pkl".format(folder_path)):
        weights = pickle_load("{}/weights.pkl".format(folder_path))
    skf = pickle_load("{}/skf.pkl".format(folder_path))

    ########################
    #####    MODELS    #####
    ########################
    
    for enum, model in enumerate(models_to_run):

        model.name = "{}_{}".format(model.name, enum)
        model.model_folder = "{}_{}".format(model.model_folder, enum)

        model_name = model.name
        model_folder = model.model_folder
        model_date = model.date

        model.dataset = 'train'

        num_class = 1

        # stacking
        X_stack_train = np.zeros((train_shape[0], num_class))
        if test_shape is not None:
            X_stack_test = np.zeros((test_shape[0], n_folds*num_class))

        RESULTS = {}
        scores_train = []
        scores_cv = []

        print_.to_print('')
        print_.to_print(model_name)
        print_.to_print(model)
        print_.title()
        
        start_0 = datetime.now()
        for fold, (train_index, cv_index) in enumerate(skf):
            
            start = datetime.now()
            
            model.fold = fold

            # load y, weights, train
            y_train, y_cv = model.load_target()
            w_train, w_cv = model.load_weights()
            X_train, X_cv = model.load_train()

            # train
            model.train(X_train, X_cv, y_train, y_cv)

            end = datetime.now()

            # make prediction
            predict_train = model.predict(X_train)
            predict_cv = model.predict(X_cv, cv=True)

            # if num_class == 1, reshape dim
            if num_class <= 1:
                predict_train = predict_train.reshape(-1, 1)
                predict_cv = predict_cv.reshape(-1, 1)

            # filling train stacking dataset
            X_stack_train[cv_index, :] = predict_cv

            # evaluating model
            train_error = metric(y_train, predict_train, weight=w_train)
            cv_error = metric(y_cv, predict_cv, weight=w_cv)
            scores_train.append(train_error)
            scores_cv.append(cv_error)

            print_.score(fold, train_error, cv_error, start, end)

            RESULTS[fold] = {
                'model': model.get_model,
                'index': [train_index, cv_index],
                'proba': [predict_train, predict_cv],
                'error': [train_error, cv_error],
            }

        end_0 = datetime.now()
        score = metric(y_true, X_stack_train, weight=weights)
        mean_train = np.mean(scores_train)
        mean_cv = np.mean(scores_cv)
        stats = [np.std(scores_cv), np.var(scores_cv)] 
        diff = score - np.mean(scores_cv)

        # print infos
        print_.line()
        print_.score('', mean_train, mean_cv, start_0, end_0)
        print_.line()

        print_.to_print("SCORE FULL TRAIN : {:.5f}".format(score))
        print_.to_print(("CV STATS :  Std : {:.5f}, "
                         "Var : {:.5f}").format(*stats))
        print_.to_print("Diff FULL TRAIN  - MEAN CV: {:.5f}".format(diff))
        print_.to_print('')

        RESULTS['train_stack_error'] = score
        RESULTS['train_stack'] = X_stack_train

        #####################################
        #####    PREDICTIONS ON TEST    #####
        #####################################

        X_test = model.load_test()

        for fold in range(n_folds):
            model_test = RESULTS[fold]['model']

            # evaluating model
            if 'xgboost' in str(model_test):
                ntree_limit = model_test.best_ntree_limit
                predict_test = model_test.predict(X_test, ntree_limit=ntree_limit)
            else:
                predict_test = model_test.predict_proba(X_test)

            # filling test stacking dataset
            if num_class <= 1:
                X_stack_test[:, fold] = predict_test
            else:
                index = np.array(list(range(num_class)))
                X_stack_test[:, index + fold*num_class] = predict_test
        
        # averaging folds
        if num_class <= 1:
            prediction = np.mean(X_stack_test, axis=1)
        else:            
            prediction = np.zeros((test_shape[0], num_class))
            for x in range(num_class):
                prediction[:, x] = X_stack_test[:, x::num_class].mean(axis=1)

        RESULTS['test_stack'] = prediction
        pickle_dump(RESULTS, "{}/{}_dump.pkl".format(model_folder, model_name))

        if do_submit:
            args = [folder_path, model_name, model_date, score]
            custom_submit(id_test, prediction, args)
    
    print_.to_print("Done ;)")
    
if __name__ == "__main__":
    main()

