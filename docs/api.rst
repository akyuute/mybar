.. module:: mybar

API Reference
===============
The following section outlines **mybar**'s Python API.


mybar.run()
------------
If you're using Python and just want a continuous status bar printed to your
screen, this function does the trick. It reads the default config file to
determine its options.

.. autofunction:: mybar.run


Fields
-------
Fields are at the core of **mybar**. They are responsible for generating
specific, meaningful, dynamic text at every iteration of a bar's lifespan.

.. autoclass:: mybar.Field
   :members:
   :exclude-members: __init__


Bars
-----
Bars control how **mybar** renders text and orchestrate all the parts that make
that happen.

.. autoclass:: mybar.Bar
   :members:
   :undoc-members:
   :exclude-members: __init__


Field Functions
----------------
Field Functions are evaluated at each `interval` to create content for Fields_.

.. automodule:: mybar.field_funcs


Containing Configs
-------------------
The configuration of a :class:`mybar.Bar` can be stored in a BarConfig, a
subclass of :class:`dict`.
This class also defines methods for making new configs from files and the
command line as well as for writing preexisting configs out to files.

.. autoclass:: mybar.BarConfig
   :show-inheritance:
   :members:
   :undoc-members:
   :exclude-members: __init__


Command Line Tools
-------------------
Use these tools to gather and prompt for data from the command line.

.. automodule:: mybar.cli
   :members:
   :exclude-members: __init__


Constants
----------
**mybar** defines the following constant values.

.. automodule:: mybar.constants
   :members:
   :undoc-members:


String Formatting
------------------
Classes for parsing, structuring and manipulating text.

.. automodule:: mybar.formatting
   :show-inheritance:
   :members:
   :exclude-members: __init__, conversions_to_secs


Config File Parsing
--------------------
**mybar** uses a custom lexer and syntax parser to gather, manipulate and
interpret data from config files.

.. automodule:: mybar.parse_conf
   :show-inheritance:
   :members:
   :undoc-members:
   :exclude-members: __init__


Utilities
----------
A miscellany of utility functions.

.. automodule:: mybar.utils
   :members:
   :exclude-members: __init__


Exceptions
-----------
A group of custom errors.

.. automodule:: mybar.errors
   :show-inheritance:
   :members:
   :exclude-members: __init__
   :undoc-members:


Custom Types
-------------
A horde of descriptive type aliases and custom data structures.

.. automodule:: mybar._types
   :show-inheritance:
   :members:
   :undoc-members:

.. automodule:: mybar.namespaces
   :show-inheritance:
   :members:
   :undoc-members:


