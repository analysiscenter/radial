"""
Radial is a library that provides a data science approach to finding the exit point on the radial mode.
"""
import re
from setuptools import setup, find_packages

with open('radial/__init__.py', 'r') as f:
    VERSION = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

setup(
    name='radial',
    packages=find_packages(exclude=['examples', 'research']),
    version=VERSION,
    url='https://github.com/analysiscenter/radial',
    author='Data Analysis Center team',
    description='A framework for constructing neural network solutions\
                 to the problem of finding the exit point of the well to the radial mode',
    zip_safe=False,
    platforms='any',
    install_requires=[
        'numpy>=1.15.4',
        'scikit-learn>=0.20.0',
        'matplotlib>=3.0.1',
        'dill>=0.2.8.2',
        'xlrd>=1.1.0',
        'pandas>=0.23.4',
        'seaborn>=0.9.0',
        'tqdm>=4.25.0',
        'tabulate>=0.8.2',
        'argparse>=1.1',
        'pytest>=4.0.0'
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.12'],
        'tensorflow-gpu': ['tensorflow-gpu>=1.12'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering'
    ],
)
