from os.path import dirname, join
from setuptools import setup, find_packages


with open(join(dirname(__file__), 'mlproject/VERSION'), 'rb') as f:
    version = f.read().decode('ascii').strip()


setup(
    name='mlproject',
    version=version,
    # url='',
    description='A high-level framework for Machine Learning project',
    long_description=open('README.rst').read(),
    author='Alexandre Araujo',
    maintainer='Alexandre Araujo',
    maintainer_email='aaraujo001@gmail.com',
    license='MIT',
    packages=find_packages(exclude=('docs', 'tests')),
    include_package_data=True,
    zip_safe=False,
    
    entry_points={
        'console_scripts': ['mlproject = mlproject.cmdline:execute']
    },

    classifiers=[
        'Framework :: mlproject',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    
    package_data={
        'data': ['generate.py.tmpl', 'train.py.tmpl','parameters.py.tmpl'],
    },

    install_requires=[

    ],
)