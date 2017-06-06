
import sys
import inspect
import argparse
from os import getcwd
from os.path import exists, join
from importlib import import_module
from pkgutil import iter_modules

import mlproject
from mlproject.commands import MlprojectCommand
from mlproject.utils import find_project_file
from mlproject.utils import inside_project


def _get_project_settings(inproject):
    if inproject:
        with open(find_project_file(getcwd()), "rb") as f:
            settings = f.read().decode("ascii").strip().split(" - ")
        return dict(zip(['name', 'date'], settings[1:]))
    return None

def _print_header(settings, inproject):
    if inproject:
        print("mlproject {} - project: {}, created: {}\n".format(\
            mlproject.__version__, settings['name'], settings['date']))
    else:
        print("mlproject {} - no active project".format(mlproject.__version__))

def _walk_modules(path):
    mods = []
    mod = import_module(path)
    mods.append(mod)
    if hasattr(mod, '__path__'):
        for _, subpath, ispkg in iter_modules(mod.__path__):
            fullpath = path + '.' + subpath
            if ispkg:
                mods += _walk_modules(fullpath)
            else:
                submod = import_module(fullpath)
                mods.append(submod)
    return mods

def _get_commands(module_name):
    cmds = {}
    for module in _walk_modules(module_name):
        for obj in vars(module).values():
            if inspect.isclass(obj) and \
                            issubclass(obj, MlprojectCommand) and \
                            obj.__module__ == module.__name__ and \
                            not obj == MlprojectCommand:
                name = module.__name__.split('.')[-1]
                cmds[name] = obj()
    return cmds

def _print_commands(cmds, settings, inproject):
    _print_header(settings, inproject)
    print("Usage:")
    print("  mlproject <command> [options] [args]\n")
    print("Available commands:")
    
    for cmdname, cmdclass in sorted(cmds.items()):
        if inproject == cmdclass.requires_project:
            print("  {:<13}{}".format(cmdname, cmdclass.short_desc()))
    if not inproject:
        print("\n\t[ more ]\tMore commands available when run from project"\
                 " directory")
    print('\nUse "scrapy <command> -h" to see more info about a command')

def _print_unknown_command(cmdname, settings, inproject):
    _print_header(settings, inproject)
    print("Unknown command: {}\n".format(cmdname))
    print('Use "mlproject" to see available commands')

def _pop_command_name(argv):
    i = 0
    for arg in argv[1:]:
        if not arg.startswith('-'):
            del argv[i]
            return arg
        i += 1

def execute():
    argv = sys.argv
    inproject = inside_project(getcwd())
    settings = _get_project_settings(inproject)
    cmds = _get_commands('mlproject.commands')

    cmdname = _pop_command_name(argv)
    if not cmdname:
        _print_commands(cmds, settings, inproject)
        sys.exit(0)
    elif cmdname not in cmds.keys():
        _print_unknown_command(cmdname, settings, inproject)
        sys.exit(2)

    cmd = cmds[cmdname]
    parser = argparse.ArgumentParser(prog=cmdname, 
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.usage = "mlproject {} {}".format(cmdname, cmd.syntax())
    parser.description = cmd.long_desc()
    
    cmd.add_options(parser)
    args = parser.parse_args()
    cmd.run(args)


if __name__ == '__main__':
    execute()