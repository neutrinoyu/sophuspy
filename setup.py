from setuptools import setup, find_packages

setup(
    name='sophuspy',
    version='0.1.0',
    author="neutinoyu",
    description="This is a pure python implementation of Lie Groups using numpy.",
    packages=find_packages(include=['sophuspy']),
    install_requires=['numpy']
)