from os import getcwd, makedirs
from os.path import join, exists
from sys import exit

import mlproject
from mlproject.commands import MlprojectCommand
from mlproject.api import GenerateWrapper
from mlproject.utils import ProgressTable, print_and_log, init_log
from mlproject.utils import print_and_log as print_
from mlproject.utils import current_folder
from mlproject.utils import Timer

"""
    this file use the GenerateWrapper API 
"""

# XXX : print fold X / nb X
# XXX : Check if dataset got object features
# XXX : time dataset generation for train and test
# XXX : print info about nan values
# XXX : print info about size (Go) of model_folder
# XXX : possiliboty de run the training after the generation process
# XXX : option to clean up after succesfull training

class Command(MlprojectCommand):

    requires_project = True

    def __init__(self):

        self.path = getcwd()
        self.logger = init_log(self.path)

    def syntax(self):
        return "[formats]"

    def short_desc(self):
        return "generate and save dataset for training"

    def add_options(self, parser):

        choices = ['xgb', 'npz', 'libsvm', 'pkl', 'libffm', 'custom']
        parser.add_argument(dest='types', choices=choices, nargs='+',
                            help='Dataset type(s) to generate') 
        parser.add_argument("--train", dest="train", action="store_true",
                        help="run training after generation")

    def _inside_code_folder(self, path):
        if not current_folder(path) == "code":
            print_(self.logger, "this command needs to be "\
                            "executed from the code folder")
        

    def _load_generate_func(self):
        from dataset import params, create_dataset, validation_splits
        self.params = params
        self.create_dataset = create_dataset
        self.validation_splits = validation_splits

    def _extract_args(self, args):
        """
        """
        self.types = args.types

    def _save_train_fold(self, df_train, validation):
        """
            save train folds
        """
        # loop over validation index
        nb_folds = len(validation)
        for fold, (train_index, cv_index) in enumerate(validation):

            self.gen.train_index, self.gen.cv_index = train_index, cv_index
            y_train, y_cv = self.gen.split_target()
            w_train, w_cv = self.gen.split_weight()
            X_train, X_cv = self.gen.split_data(df_train)

            for type_ in self.types:
                self.gen.dump(X_train, y_train, fold, w_train , type_, 'train')
                self.gen.dump(X_cv, y_cv, fold, w_cv , type_, 'cv')

            message = ('Fold {}/{}\tTrain shape\t[{}|{}]\tCV shape\t[{}|{}]')
            args = [fold, nb_folds, *X_train.shape, *X_cv.shape]
            print_(self.logger, message.format(*args))

    def _save_test(self, df_test):
        """
        """
        for type_ in self.types:
            self.gen.dump(df_test, None, None, None, type_, 'test')
        print_(self.logger, '\tTest shape\t[{}|{}]'.format(*df_test.shape))

    def run(self, args):
        """
            Generate dataset for training
        """
        if not self._inside_project(self.path): return
        if not self._inside_code_folder(self.path): return
        self._load_generate_func()
        self._extract_args(args)

        self.gen = GenerateWrapper(**self.params)

        # XXX no need => only for training step
        # # init pprint class
        # self.progess = ProgressTable(self.gen.log)


        # XXX : try to load target based on params.target
        #       if can not load push a comment and run create_dataset without split

        # XXX : if validation before creating dataset need to extract first
        # XXX : if params.validaton is a path => load file and generate validation index
        # XXX : find a way to load the target
        self.gen.validation = self.validation_splits(self.gen.n_folds, 
                                                        self.gen.y_true)

        # Generate df_train & df_test
        print_(self.logger, '\nMaking train/test dataset')
        with Timer() as t:
            df_train, df_test = self.create_dataset(self.gen.dataset, 
                                                    self.gen.validation)
        print_(self.logger, 'train/test set done in {:.0}'.format(t.interval))

        # XXX : df_train and df_test done
        # what about making the conformity test now 
        # detect nan / inf value and raise error if type format not compatible

        # extract target features
        self.gen.y_true = df_train[self.params.target_name].values
        
        # clean dataset
        df_train = self.gen.cleaning(df_train)
        df_test = self.gen.cleaning(df_test)

        # save infos
        self.gen.get_train_infos(df_train)
        self.gen.get_test_infos(df_test)

        # save and generate features map
        self.gen.create_feature_map()

        # save train fold
        self._save_train_fold(df_train, self.gen.validation)

        # save test and do conformity tests 
        # between train and test
        if df_test is not None:
            self._save_test(df_test)
            self.gen.conformity_test()

        # save infos
        self.gen.save_infos()
        self.gen.copy_script()