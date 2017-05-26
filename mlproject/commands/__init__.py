"""
Base class for mlproject commands
"""
from mlproject.utils import inside_project

class MlprojectCommand(object):

    def syntax(self):
        """
            Command syntax (preferably one-line). Do not include command name.
        """
        return ""

    def short_desc(self):
        """
            A short description of the command
        """
        return ""

    def long_desc(self):
        """
            A long description of the command. Return short description when 
            not available. It cannot contain newlines, since contents will be 
            formatted by optparser which removes newlines and wraps text.
        """
        return self.short_desc()

    def help(self):
        """
            An extensive help for the command. It will be shown when using the
            "help" command. It can contain newlines, since not post-formatting 
            will be applied to its contents.
        """
        return self.long_desc()

    def add_options(self, parser):
        """
            add options to the arguments parser
        """
        pass

    def check_inside_project(self, path):
        inproject = inside_project(path)
        if not inproject:
            exit("command needs to be run from inside a project")

    def run(self, args, opts):
        """
            Entry point for running commands
        """
        raise NotImplementedError