.. mybar documentation master file, created by
   sphinx-quickstart on Mon Dec 26 22:54:57 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

######
mybar
######

*Craft highly customizable status bars with ease.*


About mybar
------------

Introduction
=============

**mybar** is a code library and command line tool written in Python for making
status bars.

It aims to aid users in creating custom status bars with intuitive
controls that allow for the customization of every element.

::

   $ python -m mybar --template '{uptime} [{cpu_usage}/{cpu_temp}] | {battery}'
   Up 4d:14h:19m [CPU 03%/36C] | Bat 100CHG



Install mybar
--------------

**mybar** supports Python 3.11+.

It can be installed from the `Python Package Index`::

   $ python -m pip install mybar



Use mybar in the command line
------------------------------

By default, **mybar** looks at config files to load its options.

Running the **mybar** command line tool using your default config file is as simple as::

   $ python -m mybar


The first time you run **mybar**, it will check if you have a config file in the default location::

   -- mybar --
   The default config file at '/home/me/.config/mybar/mybar.conf' does not exist.
   Would you like to make it now? [Y/n] y
   Wrote new config file to '/home/me/.config/mybar/mybar.conf'

You can also skip writing a config file and **mybar** will start running a bar with default
parameters::

   Would you like to make it now? [Y/n] n
   mymachine|Up 4d:14h:54m|CPU 04%|36C|Mem 3.8G|/:50.8G|Bat 100CHG|wifi|2023-08-01 17:06:04

Note that any options passed to the command on the first run will be written to the new config file.


Command line examples
**********************

Let's see some examples of how to use **mybar** from the command line.


``--fields`` Specify which fields to show:

.. code:: bash

   $ python -m mybar --fields hostname disk_usage cpu_temp datetime
   mymachine|/:88.3G|43C|2023-08-01 23:18:22


``--template`` Use a custom format template:

.. code:: bash

   $ python -m mybar --template '@{hostname}: ( {uptime} | {cpu_usage}, {cpu_temp} )  [{datetime}]'
   @mymachine: ( Up 1d:12h:17m | CPU 02%, 44C )  [2023-08-01 23:31:26]


``--separator`` Change the field separator:

.. code:: bash

   $ python -m mybar -f hostname uptime cpu_usage --separator ' ][ '
   mymachine ][ Up 1d:12h:11m ][ CPU 00%


``--refresh`` Set the bar's refresh rate:

.. code:: bash

   $ python -m mybar --refresh 5


``--count`` Run the bar a specific number of times:

.. code:: bash

   $ python -m mybar -f hostname cpu_usage datetime --count 3 --endline
   mymachine|CPU 00%|2023-08-01 23:40:26
   mymachine|CPU 00%|2023-08-01 23:40:27
   mymachine|CPU 00%|2023-08-01 23:40:28
   $


``--icons`` Set new icons for each field:

.. code:: bash

   $ python -m mybar -f hostname cpu_usage datetime --icons cpu_usage='@' datetime='Time: '
   mymachine|@03%|Time: 2023-08-02 01:01:56


``--options`` Set arbitrary options for the bar or any field:

.. code:: bash

   $ python -m mybar -t '@{hostname} {cpu_usage} Time: {datetime}' --options datetime.kwargs.fmt='%H:%M:%S.%f'
   @mymachine CPU 00% Time: 01:19:55.000229


``--config`` Use a specific config file:

.. code:: bash

   $ python -m mybar --config ~/.config/mybar/my_other_config_file.conf


See the `manual` for details on all the command line arguments **mybar** accepts.



Use mybar in a Python project
------------------------------

>>> import mybar

See `docs.api.rst` for in-depth Python API usage.

Python API examples
********************

Let's see some examples of how to use **mybar** using the Python API.

Get started with some default Fields:

.. code:: python

   >>> some_default_fields = ['uptime', 'cpu_temp', 'battery', 'datetime']
   >>> sep = ' ][ '
   >>> using_defaults = mybar.Bar(fields=some_default_fields, separator=sep)
   >>> using_defaults
   Bar(fields=['uptime', 'cpu_temp', 'battery', ...])
   >>> using_defaults.run()
   Up 1d:10h:31m ][ 43C ][ Bat 100CHG ][ 2023-08-01 21:43:40


Load a Bar from a config file:

.. code:: python

   >>> mybar.Bar.from_file('~/mycustombar.conf')
   Bar(fields=['hostname', 'custom_field1', 'disk_usage', ...])


Use your own functions to bring your Bar to life:

.. code:: python

   >>> def database_reader(query: str) -> str:
           return read_from_database(query)

   >>> my_field = mybar.Field(func=database_reader, kwargs={'query': '...'}, interval=60)
   >>> my_field
   Field(name='database_reader')
   >>> bar = mybar.Bar(fields=[my_field, 'hostname', 'datetime'], refresh_rate=2)


Append new Fields to your Bar, as if it were a list:

.. code:: python

   >>> bar.fields
   (Field(name='database_reader'), Field(name='hostname'), Field(name='datetime'))
   >>> bar.append(Field.from_default('uptime'))
   Bar(fields=['database_reader', 'hostname', 'datetime', ...])
   >>> bar.fields
   (Field(name='database_reader'), Field(name='hostname'), Field(name='datetime'), Field(name='uptime'))



Concepts
---------

This section introduces the core concepts that aid in customizing **mybar**.

- Bar
      The status bar.
- Field
      A part of the `Bar` containing information, sometimes called a "module"
      by other status bar frameworks.
- field function
      The function a `Field` runs to determine what it should contain.
- refresh cycle
      The time it takes the `Bar` to run all its fields and update its contents once.
- refresh rate
      How often the `Bar` updates what it says, in seconds per refresh.
- interval
      How often a `Field` runs its field function, in seconds per cycle.
- separator
      A string that separates one `Field` from another
- format string
      A special string that controls how `Fields` and their contents are displayed.
- icon
      A string appearing with each `Field`, usually unique to each.


To customize **mybar** to your liking without using the `Python API`, you can use `config files`
or `command line arguments`.


.. Configuration Files
.. ====================


.. Advanced Usage // Field Funcs
.. ============

.. `Field funcs` are Python functions that return the contents of a `Field`.

.. Read more about them in `docs.api.rst`.



Default Fields
---------------

These are the default fields in mybar.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   cli
   usage
   bars
   fields
   configuration
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
