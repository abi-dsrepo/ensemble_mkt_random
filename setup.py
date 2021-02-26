from setuptools import setup, find_packages
import src.__meta__ as meta

meta_app = meta.__app__
meta_version = meta.__version__
meta_description = meta.__description__

PACKAGE_NAME: str = 'src'

setup(
    name=meta_app,
    version=meta_version,
    author="Abinaya",
    author_email="jabinaya.29@gmail.com",
    packages=find_packages(
        include=[PACKAGE_NAME, f'{PACKAGE_NAME}.*'],
        exclude=("tests",)
    ),
    include_package_data = True,
    package_data={
        # If any package contains *.ini files or json, include them
        '': ['*.json'],
    },
    tests_require=["pytest"]
)
