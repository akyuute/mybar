from setuptools import setup

from mybar import (
    __title__,
    __description__,
    __url__,
    __version__,
    __author__,
    __license__,
    __copyright__,
)

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

kwargs = dict(

##    version=VERSION,
##    description=(
##        "A lightweight, highly customizable asynchronous status bar API "
##        "for terminal and GUI environments."
##    ),
##    author='derek-yeetr',
##    url='https://github.com/derek-yeetr/mybar',
##    license='MIT',

    name='mybar',
    version=__version__,
    description=__description__,
    long_description=readme,
    long_description_content_type='text/markdown',
    author=__author__,
    url=__url__,
    packages=[
        'mybar'
    ],
    classifiers=[

        "Development Status :: 3 :: Alpha",
        "Environment :: Console",
        "Framework :: AsyncIO",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        # "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Desktop Environment",
        "Topic :: Utilities",

    ],

    license=__license__,
    # license_files="LICENSE",
    # keywords=[
    # ],

    platforms="Platform Independent",

    install_requires=requirements,
    python_requires='>=3.10',

    project_urls={
        'Source': "https://github.com/derek-yeetr/mybar",
    }
)

setup(**kwargs)

