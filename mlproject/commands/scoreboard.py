

from mlproject.commands import MlprojectCommand

class Command(MlprojectCommand):

    requires_project = True

    def syntax(self):
        return ""

    def short_desc(self):
        return "print scoreboard"

    def run(self, args):
        """
            Fetch generate file from model directory
        """
        print(args)
        print('run fetch')