
import unittest
import glob
from os.path import exists, join
from tempfile import TemporaryDirectory
from subprocess import Popen, PIPE
from itertools import product


class CmdlineTests(unittest.TestCase):

    def _execute(self, cmd):
        process = Popen(cmd, stdout=PIPE, bufsize=1, shell=True).communicate()
        return process

    def test_help(self):
        cmd = "mlproject"
        process = self._execute(cmd)

    def test_starproject(self):
        with TemporaryDirectory() as path:
            path = join(path, 'ml_project')
            cmd = "mlproject startproject {}".format(path)
            process = self._execute(cmd)

            self.assertTrue(exists(join(path, 'code')))
            self.assertTrue(exists(join(path, 'models')))
            self.assertTrue(exists(join(path, 'data')))
            self.assertTrue(exists(join(path, 'jupyter')))
            for folder in ['binary', 'raw', 'features']:
                self.assertTrue(exists(join(path, 'data', 'train', folder)))
                self.assertTrue(exists(join(path, 'data', 'test', folder)))

    def test_generate(self):
        with TemporaryDirectory() as path:

            # XXX : import this infos
            extensions = ['xgb', 'npz', 'pkl']

            path = join(path, 'ml_project')
            cmd = "mlproject startproject {} --test_code;".format(path)
            cmd += "cd {};".format(join(path, "code"))
            cmd += "mlproject generate {};".format(' '.join(extensions))
            process = self._execute(cmd)

            folders = glob.glob(join(path, "models", "**"))
            self.assertTrue(len(folders))

            path = join(path, "models", folders[0])
            gen = product(range(5), extensions, ['tr', 'va'])
            for fold, ext, set_ in gen:
                self.assertTrue(exists(join(path, "fold_{}".format(fold), 
                    "X_{}.{}".format(set_, ext))))
            for ext in extensions:
                self.assertTrue(exists(join(path, 
                        "test", "X_test.{}".format(ext))))


if __name__ == '__main__':
    unittest.main()