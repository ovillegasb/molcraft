#!/usr/bin/env python

"""
This script helps to install the molcraft module correctly.

    Package to tar.gz:
        python setup.py sdist

    Install:
        python setup.py install

"""

from distutils.core import setup

setup(
    name="molcraft",
    version="0.1.0",
    description="Python utilities for working with molecular systems",
    author="Orlando Villegas",
    author_email="ovillegas.bello0317@gmail.com",
    url="https://github.com/ovillegasb/molcraft",
    packages=["molcraft"],  # package_dir={"molcraft": "./molcraft"},
    package_data={"molcraft": ["ffdata/*.dat"]},
    py_modules=["molcraft.structure"],  # requires=[""]
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities"  
        # https://docs.python.org/es/3/distutils/setupscript.html
    ],
    keywords=["Chemistry"]
)
