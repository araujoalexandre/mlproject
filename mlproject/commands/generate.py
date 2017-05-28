from os import getcwd, makedirs
from os.path import join, exists
from sys import exit

import mlproject
from mlproject.commands import MlprojectCommand
from mlproject.api import GenerateWrapper
from mlproject.utils import pprint
from mlproject.utils import current_folder
from mlproject.utils import Timer

# XXX : print fold X / nb X
# XXX : Check if dataset got object features
# XXX : time dataset generation for train and test
# XXX : print info about nan values
# XXX : print info about size (Go) of model_folder
# XXX : possiliboty de run the training after the generation process
# XXX : option to clean up after succesfull training

class Command(MlprojectCommand):

    requires_project = True

    def syntax(self):
        return ""

    def short_desc(self):
        return "generate dataset for training"

    def add_options(self, parser):

        choices = ['xgb', 'npz', 'libsvm', 'pkl', 'libffm', 'custom']
        parser.add_argument('-t','--type', help='Dataset type(s) to generate', 
                                           nargs='+',
                                           choices=choices,
                                           dest='types',
                                           required=True)
        parser.add_argument("--train", dest="train", action="store_true",
                        help="run training after generation")

    def _check_folder(self):
        path = getcwd()
        if not current_folder(path) == "code":
            print("this command needs to be executed from the code folder")
            exit(0)

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
            self.print_.to_print(message.format(*args))

    def _save_test(self, df_test):
        """
        """
        for type_ in self.types:
            self.gen.dump(df_test, type_, 'test')
        self.print_.to_print('\tTest shape\t[{}|{}]'.format(*df_test.shape))

    def run(self, args):
        """
            Generate dataset for training
        """
        self.check_inside_project(getcwd())
        self._check_folder()
        self._load_generate_func()
        self._extract_args(args)

        self.gen = GenerateWrapper(**self.params)

        # init pprint class
        self.print_ = pprint(self.gen.log)

        # XXX : if validation before creating dataset need to extract first
        # XXX : if params.validaton is a path => load file and generate validation index
        # XXX : find a way to load the target
        self.gen.validation = self.validation_splits(self.gen.n_folds, 
                                                        self.gen.y_true)

        # Generate df_train & df_test
        self.print_.to_print('\nMaking train/test dataset')
        with Timer() as t:
            df_train, df_test = self.create_dataset(self.gen.dataset, 
                                                    self.gen.validation)
        self.print_.to_print('train/test set done in {:.0}'.format(t.interval))

        # extract target features
        self.gen.y_true = df_train[self.params.target_name].values
        
        # clean dataset
        df_train = self.gen.cleaning(df_train)
        # save infos
        self.gen.get_infos(df_train)
        # save and generate features map
        self.gen.create_feature_map()
        # save train fold
        self._save_train_fold(df_train, self.gen.validation)

        # XXX : add condition if not test
        if do_test:

            # clean dataset
            df_test = self.gen.cleaning(df_test)
            # save infos
            self.gen.get_infos(df_test)
            # save test
            self._save_test(df_test)
            # do conformity tests
            self.gen.conformity_test()

        # save infos
        self.gen.save_infos()
        self.gen.copy_script()