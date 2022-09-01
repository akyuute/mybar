from setuptools import setup


VERSION = '0.1.0'


readme = ""
with open('README.md') as f:
    readme = f.read()

requirements = []
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='mybar',
    author='derek-yeetr',
    url='https://github.com/derek-yeetr/mybar',
    version=VERSION,
    packages=[
        'mybar'
    ],
    description=(
        "A lightweight, highly customizable asynchronous status bar API "
        "for terminal and GUI environments."
    ),
    long_description=readme,
    install_requires=requirements,

)


