.. module:: mybar

API Reference
===============

The following section outlines the API of mybar.


.. autofunction:: run


Constants
----------

.. code:: py

    CONFIG_FILE: str = '~/.mybar.json'
    # Unix terminal escape code (control sequence introducer):
    CSI: ConsoleControlCode = '\033['
    CLEAR_LINE: ConsoleControlCode = '\x1b[2K'  # VT100 escape code to clear line
    HIDE_CURSOR: ConsoleControlCode = '?25l'
    UNHIDE_CURSOR: ConsoleControlCode = '?25h'




Bars
-----

.. autoclass:: Bar
   :members:
   :exclude-members: __init__

.. autoclass:: Template
   :exclude-members: __init__
   :members:


Fields
-------

.. autoclass:: Field
   :members:
   :exclude-members: __init__


Command Line Tools
-------------------

.. automodule:: mybar.cli
   :exclude-members: __init__
   :members:


Exceptions
-----------

.. automodule:: mybar.errors
   :members:

autoexception


Custom Types
-------------

.. automodule:: mybar.types
   :members:


