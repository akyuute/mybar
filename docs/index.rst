..
    .. meta::
       :title: mybar Documentation
       :description: Documentation for mybar, the one tool for superb status bars.
       :language: en-US
       :keywords: mybar, python, status bar
       :copyright: 2021-present akyuute


.. image:: _static/logo.png
   :align: center
   :alt: Logo


######
mybar
######

*Craft highly customizable status bars with ease.*



Introduction
=============

**mybar** is a code library and command line tool written in Python for making
status bars.

It aims to aid users in creating custom status bars with intuitive
controls that allow for the customization of every element.

.. code:: bash

   $ python -m mybar --template '{uptime} [{cpu_usage}/{cpu_temp}] | {battery}'
   Up 4d:14h:19m [CPU 03%/36C] | Bat 100CHG



Install mybar
==============

**mybar** supports Python 3.12+.

It can be installed from the `Python Package Index <https://pypi.org/project/mybar/>`_:

.. code:: bash

   $ python -m pip install mybar



Use mybar in the command line
==============================

By default, **mybar** looks at config files to load its options.
**mybar** uses the `Scuff <https://github.com/akyuute/scuff>`_
language to process data from and write config files.

The default config file location is ``~/.config/mybar/mybar.conf``

Running the **mybar** command line tool using your default config file is as simple as:

.. code:: bash

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
~~~~~~~~~~~~~~~~~~~~~~

See :doc:`cli` for details on all the command line arguments **mybar** accepts.

Let's see some examples of how to use **mybar** from the command line.


``--fields/-f`` Specify which Fields to show:

.. code:: bash

   $ python -m mybar -f hostname disk_usage cpu_temp datetime
   mymachine|/:88.3G|43C|2023-08-01 23:18:22


``--template/-t`` Use a custom format template:

.. code:: bash

   $ python -m mybar -t '@{hostname}: ({uptime} | {cpu_usage}, {cpu_temp})  [{datetime}]'
   @mymachine: (Up 1d:12h:17m | CPU 02%, 44C)  [2023-08-01 23:31:26]


``--separator/-s`` Change the Field separator:

.. code:: bash

   $ python -m mybar -f hostname uptime cpu_usage -s ' ][ '
   mymachine ][ Up 1d:12h:11m ][ CPU 00%


``--count/-n`` Run the Bar a specific number of times:

.. code:: bash

   $ python -m mybar -f hostname cpu_usage datetime -n 3 --break-lines
   mymachine|CPU 00%|2023-08-01 23:40:26
   mymachine|CPU 00%|2023-08-01 23:40:27
   mymachine|CPU 00%|2023-08-01 23:40:28
   $


``--refresh/-r`` Set the Bar's refresh rate:

.. code:: bash

   $ python -m mybar -f hostname cpu_usage datetime -n 3 -r 10 --break-lines
   mymachine|CPU 00%|2023-11-24 04:25:31
   mymachine|CPU 00%|2023-11-24 04:25:41
   mymachine|CPU 00%|2023-11-24 04:25:51
   $


``--icons/-i`` Set new icons for each Field:

.. code:: bash

   $ python -m mybar -i uptime='⏱️' cpu_temp='🔥' mem_usage='🧠' battery='🔋'
   mymachine|⏱️4d:15h:7m|CPU 00%|🔥50C|🧠8.7G|/:80.7G|🔋100CHG|wifi|2023-11-10 17:19:20


``--options/-o`` Set arbitrary options for the bar or any Field:

.. code:: bash

   $ python -m mybar -t 'Time: {datetime}' -o datetime.kwargs.fmt='%H:%M:%S.%f'
   Time: 01:19:55.000229


``--config/-c`` Use a specific config file:

.. code:: bash

   $ python -m mybar -c ~/.config/mybar/my_other_config_file.conf



Use mybar in a Python project
==============================

.. code:: python

    >>> import mybar


Python API examples
~~~~~~~~~~~~~~~~~~~~

See :doc:`api` for in-depth Python API documentation.

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


To customize **mybar** to your liking without using the `Python API`,
you can use :doc:`config file options <configuration>` or :doc:`command line arguments <cli>`.



Concepts
=========

This section introduces the core concepts that aid in customizing **mybar**.

- *Bar*
      The status bar.
- *Field*
      A part of the `Bar` containing information, often called a "module"
      by other status bar frameworks.
- *Field function*
      The function a `Field` runs to determine what it should contain.
- *Refresh rate*
      How often the `Bar` updates what it says, in seconds.
- *Interval*
      How often in seconds `Field` contents are updated.
      This can differ from the `refresh rate`,
      but generally should not be smaller than it.
- *Template*
      A template string that specifies how a `Bar` should be formatted.
      `Field` names put in curly brackets (``{}``) are replaced by their
      dynamically updated contents. Individual `Fields` can also have
      their own `templates`.
- *Separator*
      A string that separates each `Field` from the other in the `Bar`
      when not using a `template`.
- *Icon*
      A string appearing with each `Field`, usually unique to each.
      A `Field` using its own `template` can place it anywhere inside
      itself with the ``{icon}`` specifier.


.. `Field funcs` are Python functions that return the contents of a `Field`.
.. Read more about them in :doc:`api`


.. Default Fields
.. ---------------
.. 
.. These are the default fields in mybar.



###################
More Documentation
###################

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
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

