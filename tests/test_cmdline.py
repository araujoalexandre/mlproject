
import unittest
from os.path import exists, join
from tempfile import TemporaryDirectory
from subprocess import Popen, PIPE



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


if __name__ == '__main__':
    unittest.main()