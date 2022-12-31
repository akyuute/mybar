Command Line Usage
===================


Synopsis
---------

.. code::

    python -m mybar [ ([-m <format_string>] | [-f <fieldnames>]) ]
                    [-r <refresh_rate>] [-c <file>] [--debug]
                    [--icons <fieldname>=<icon> ...]
                    [-s <field_separator>]
                    [-o] [-j] [-h]


Options
-------

.. option:: -h, --help

    Show a help message and exit.

.. option:: -m <format_string>, --format <format_string>

    A curly-brace-delimited format string. Not valid with :option:`--fields`

.. option:: -f <fieldnames>, --fields <fieldnames>

    A list of fields to be displayed. Not valid with :option:`--format`

.. option:: -r <refresh_rate>, --refresh <refresh_rate>

    The bar's refresh rate in seconds per cycle.

.. option:: --config <file>, -c <file>

    The config file to use for default settings.

.. option:: --once, -o

    Print the bar once and exit.

.. option:: --debug

    Use debug mode.


Options for :option:`--fields`:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following options are not valid when using --format/-m:

.. option:: --icons <fieldname>=<icon> ...

    A mapping of field names to icons.

.. option:: -s <field_separator>, --separator <field_separator>

    The character used for joining fields.

.. option:: --join-empty, -j

    Include empty field contents instead of hiding them.
    Example:



Examples
---------

Run the bar using your default config file:

.. code:: bash

    $ python -m mybar
    rutherford|Up 6d:11h:20m|CPU 02%|34C|Mem 5.2G|/:99.3G|Bat CHG100|WiFi|2022-12-27 10:21:09



Run the bar using specific fields:

.. code:: bash

    $ python -m mybar -f uptime cpu_temp mem_usage
    Up 6d:11h:32m|37C|Mem 5.3G



Run the bar using a custom format string:

.. code:: bash

    $ python -m mybar -m '{uptime}! [{cpu_usage}/{cpu_temp}]; {datetime}'
    Up 6d:11h:37m! [CPU 01%/36C]; 2022-12-28 01:37:58




.. seealso::

   The :doc:`configuration` page shows more comprehensive customization options through the use of config files.

