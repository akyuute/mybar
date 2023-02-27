.. module:: mybar

API Reference
===============

The following section outlines the API of mybar.


mybar.run()
------------

.. autofunction:: run


Bars
-----

.. autoclass:: Bar
   :members:
   :undoc-members:
   :exclude-members: __init__


Fields
-------

.. autoclass:: Field
   :members:
   :exclude-members: __init__


Field Functions
----------------
Field Functions are evaluated at each `interval` to create content for Fields_.

.. automodule:: mybar.field_funcs


Templates
----------

.. autoclass:: mybar.templates.BarTemplate
   :show-inheritance:
   :members:
   :exclude-members: __init__


Command Line Tools
-------------------

.. automodule:: mybar.cli
   :members:
   :exclude-members: __init__


Constants
----------

.. automodule:: mybar.constants
   :show-inheritance:
   :members:
   :undoc-members:


Utilities
----------
.. automodule:: mybar.utils
   :members:
   :exclude-members: __init__


Exceptions
-----------

.. automodule:: mybar.errors
   :members:
   :exclude-members: __init__
   :undoc-members:

..
   autoclass:: mybar.errors.FormatStringError
   :show-inheritance:
   :members:
   :exclude-members: __init__

   autoclass:: mybar.errors.BrokenFormatStringError,
   :show-inheritance:
   :members:
   :exclude-members: __init__

   autoclass:: mybar.errors.MissingFieldnameError
   :show-inheritance:
   :members:
   :exclude-members: __init__



Custom Types
-------------

.. automodule:: mybar._types
   :show-inheritance:
   :members:
   :undoc-members:



