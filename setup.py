
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

setup(
    name="MolCraft",
    version="0.1.0",
    packages=find_packages(),
    install_requires=install_requires,
    author="Orlando Villegas",
    author_email="ovillegas.bello0317@gmail.com",
    description='Suite of tools for working with molecular systems.',
    url='https://github.com/ovillegasb/molcraft'
)
