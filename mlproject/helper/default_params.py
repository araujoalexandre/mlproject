
##########################
#####    LightGBM    #####
##########################

# 'params': {
#     'application': 'regression',
#     'num_iterations': 10,
#     'learning_rate': 0.1,
#     'num_leaves': 127,
#     'tree_learner': 'serial',
#     'num_threads': 1,
#     'min_data_in_leaf': 100,
#     'metric': 'l2',
#     'is_training_metric': False,
#     'feature_fraction': 1.,
#     'feature_fraction_seed': 2,
#     'bagging_fraction': 1.,
#     'bagging_freq': 0,
#     'bagging_seed': 3,
#     'metric_freq': 1,
#     'early_stopping_round': 0,
#     'max_bin': 255,
#     'is_unbalance': False,
#     'num_class': '1',
#     'boosting_type': 'gbdt',
#     'min_sum_hessian_in_leaf': 10,
#     'drop_rate': 0.01,
#     'drop_seed': 4,
#     'max_depth': -1,
#     'lambda_l1': 0.,
#     'lambda_l2': 0.,
#     'min_gain_to_split': 0.,
#     'verbose': True,
# },

################################################
#####    DEFAULT PARAMETERS FOR XGBoost    #####
################################################

## Parameters for booster
# 'num_boost_round': 10, 
# 'obj': None, 
# 'feval': None, 
# 'maximize': False, 
# 'early_stopping_rounds': None, 
# 'evals_result': None, 
# 'verbose_eval': True, 
# 'learning_rates': None, 
# 'xgb_model': None, 
# 'callbacks': None

## General parameters
# 'booster': 'gbtree'
# 'silent': 0,
# 'nthread': -1,

## Parameters for Tree Booster
# 'eta': 0.3,
# 'gamma': 0,
# 'max_depth': 6,
# 'min_child_weight': 1,
# 'max_delta_step': 0,
# 'subsample': 1,
# 'colsample_bytree': 1,
# 'colsample_bylevel': 1,
# 'lambda': 1,
# 'alpha': 0,
# 'sketch_eps': 0.03,
# 'scale_pos_weight': 0,

## Additional parameters for Dart Booster
# 'sample_type': 'uniform',
# 'normalize_type': 'tree',
# 'rate_drop': 0.0
# 'range': [0.0, 1.0],
# 'skip_drop': 0.0,

## Parameters for Linear Booster
# 'lambda': 0,
# 'alpha': 0,
# 'lambda_bias': 0,

## Learning Task Parameters
# 'objective': 'reg:linear',
# 'base_score': 0.5,
# 'eval_metric': [default according to objective],
# 'seed': 0,


###############################################
#####    DEFAULT PARAMETERS FOR LIBSVM    #####
###############################################

# 'svm-train' Usage
# =================
# Usage: svm-train [options] training_set_file [model_file]
# options:
# -s svm_type : set type of SVM (default 0)
#     0 -- C-SVC      (multi-class classification)
#     1 -- nu-SVC     (multi-class classification)
#     2 -- one-class SVM  
#     3 -- epsilon-SVR    (regression)
#     4 -- nu-SVR     (regression)
# -t kernel_type : set type of kernel function (default 2)
#     0 -- linear: u'*v
#     1 -- polynomial: (gamma*u'*v + coef0)^degree
#     2 -- radial basis function: exp(-gamma*|u-v|^2)
#     3 -- sigmoid: tanh(gamma*u'*v + coef0)
#     4 -- precomputed kernel (kernel values in training_set_file)
# -d degree : set degree in kernel function (default 3)
# -g gamma : set gamma in kernel function (default 1/num_features)
# -r coef0 : set coef0 in kernel function (default 0)
# -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
# -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
# -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
# -m cachesize : set cache memory size in MB (default 100)
# -e epsilon : set tolerance of termination criterion (default 0.001)
# -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
# -b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
# -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
# -v n: n-fold cross validation mode
# -q : quiet mode (no outputs)



##################################################
#####    DEFAULT PARAMETERS FOR LIBLINEAR    #####
##################################################

# -s type : set type of solver (default 1)
#   for multi-class classification
#          0 -- L2-regularized logistic regression (primal)
#          1 -- L2-regularized L2-loss support vector classification (dual)
#          2 -- L2-regularized L2-loss support vector classification (primal)
#          3 -- L2-regularized L1-loss support vector classification (dual)
#          4 -- support vector classification by Crammer and Singer
#          5 -- L1-regularized L2-loss support vector classification
#          6 -- L1-regularized logistic regression
#          7 -- L2-regularized logistic regression (dual)
#   for regression
#         11 -- L2-regularized L2-loss support vector regression (primal)
#         12 -- L2-regularized L2-loss support vector regression (dual)
#         13 -- L2-regularized L1-loss support vector regression (dual)
# -c cost : set the parameter C (default 1)
# -p epsilon : set the epsilon in loss function of SVR (default 0.1)
# -e epsilon : set tolerance of termination criterion
#         -s 0 and 2
#                 |f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,
#                 where f is the primal function and pos/neg are # of
#                 positive/negative data (default 0.01)
#         -s 11
#                 |f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)
#         -s 1, 3, 4, and 7
#                 Dual maximal violation <= eps; similar to libsvm (default 0.1)
#         -s 5 and 6
#                 |f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,
#                 where f is the primal function (default 0.01)
#         -s 12 and 13
#                 |f'(alpha)|_1 <= eps |f'(alpha0)|,
#                 where f is the dual function (default 0.1)
# -B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)
# -wi weight: weights adjust the parameter C of different classes (see README for details)
# -v n: n-fold cross validation mode
# -C : find parameter C (only for -s 0 and 2)
# -n nr_thread : parallel version with [nr_thread] threads (default 1; only for -s 0, 1, 2, 3, 11)
# -q : quiet mode (no outputs)

# `predict' Usage
# ===============
# Usage: predict [options] test_file model_file output_file
# options:
# -b probability_estimates: whether to output probability estimates, 0 or 1 (default 0); currently for logistic regression only
# -q : quiet mode (no outputs)
# Note that -b is only needed in the prediction phase. This is different
# from the setting of LIBSVM.



########################
#####    LIBFFM    #####
########################

# Command Line Usage
# ==================

# -   `ffm-train'

#     usage: ffm-train [options] training_set_file [model_file]

#     options:
#     -l <lambda>: set regularization parameter (default 0.00002)
#     -k <factor>: set number of latent factors (default 4)
#     -t <iteration>: set number of iterations (default 15)
#     -r <eta>: set learning rate (default 0.2)
#     -s <nr_threads>: set number of threads (default 1)
#     -p <path>: set path to the validation set
#     -v <fold>: set the number of folds for cross-validation
#     --quiet: quiet model (no output)
#     --no-norm: disable instance-wise normalization
#     --no-rand: disable random update
#     --on-disk: perform on-disk training (a temporary file <training_set_file>.bin will be generated)
#     --auto-stop: stop at the iteration that achieves the best validation loss (must be used with -p)

#     By default we do instance-wise normalization. That is, we normalize the 2-norm of each instance to 1. You can use
#     `--no-norm' to disable this function.
    
#     By default, our algorithm randomly select an instance for update in each inner iteration. On some datasets you may
#     want to do update in the original order. You can do it by using `--no-rand' together with `-s 1.'

#     If you do not have enough memory, then you can use `--on-disk' to do disk-level training. Two restrictions when you
#     use this mode:
        
#         1. So far we do not allow random update in the mode, so please use
#            `--no-rand' if you want to do on-disk training. 
           
#         2. Cross-validation in this mode is not yet supported.

#     A binary file `training_set_file.bin' will be generated to store the data in binary format.

#     Because FFM usually need early stopping for better test performance, we provide an option `--auto-stop' to stop at
#     the iteration that achieves the best validation loss. Note that you need to provide a validation set with `-p' when
#     you use this option.


# -   `ffm-predict'

#     usage: ffm-predict test_file model_file output_file
