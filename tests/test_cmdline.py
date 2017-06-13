import sys
import unittest
import glob
from os.path import exists, join, abspath, dirname
from shutil import copyfile
from tempfile import TemporaryDirectory, mkdtemp
from subprocess import Popen, PIPE, call
from itertools import product

from mlproject.utils import get_ext_cls


class BaseTests(unittest.TestCase):

    project_name = "proj123"

    def _execute(self, *args, **kwargs):
        cmd = (sys.executable, "-m", "mlproject.cmdline") + args
        return Popen(cmd, stdout=PIPE, stderr=PIPE, **kwargs).communicate()


class StartPojectTests(BaseTests):

    def test_starproject(self):
        with TemporaryDirectory() as path:

            p = self._execute("startproject", self.project_name, path)
            path = join(path, self.project_name)

            self.assertTrue(exists(join(path, 'code').strip()))
            self.assertTrue(exists(join(path, 'models')))
            self.assertTrue(exists(join(path, 'data')))
            self.assertTrue(exists(join(path, 'jupyter')))
            for folder in ['binary', 'raw', 'features']:
                self.assertTrue(exists(join(path, 'data', 'train', folder)))
                self.assertTrue(exists(join(path, 'data', 'test', folder)))

class GenerateTest(BaseTests):

    def setUp(self):
        self.dir = TemporaryDirectory()
        self._execute('startproject', self.project_name, self.dir.name)
        self.path = join(self.dir.name, self.project_name)

        scripts_dir = join(abspath(dirname(__file__)), 'scripts')
        # override project file in code folder
        src = join(scripts_dir, "testcase1.py")
        copyfile(src, join(self.path, "code", "project.py"))

    def tearDown(self):
        self.dir.cleanup()

    def test_generate(self):

        extensions = list(get_ext_cls().keys())
        # extensions = ['xgb', 'npz', 'pkl']
        p = self._execute('generate', *extensions, cwd=join(self.path, "code"))

        models_dirs = glob.glob(join(self.path, "models", "**"))
        self.assertTrue(len(models_dirs))

        model = join(self.path, "models", models_dirs[0])
        gen = product(range(5), extensions, ['tr', 'va'])
        for fold, ext, set_ in gen:
            self.assertTrue(exists(join(model, "fold_{}".format(fold), 
                "X_{}.{}".format(set_, ext))))
        for ext in extensions:
            self.assertTrue(exists(join(model, 
                    "test", "X_test.{}".format(ext))))

if __name__ == '__main__':
    unittest.main()