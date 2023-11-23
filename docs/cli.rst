Command Line Usage
===================


Synopsis
---------

.. code:: none

    python -m mybar [--help] [--template 'TEMPLATE' | --fields FIELDNAME1 [FIELDNAME2 ...]]
                    [--separator 'FIELD_SEPARATOR'] [--join-empty]
                    [--options FIELD1.OPTION='VAL' [FIELD2.OPTION='VAL' ...]]
                    [--refresh REFRESH] [--count TIMES] [--break-lines]
                    [--icons FIELDNAME1='ICON1' [FIELDNAME2='ICON2' ...]]
                    [--unicode] [--no-unicode] [--config FILE] [--dump] [--debug]
                    [--version]

Options
-------


.. option:: -h, --help

    show this help message and exit

.. option:: --template 'TEMPLATE', -t 'TEMPLATE'

    A curly-brace-delimited format string. Not valid with :option:`--fields` options.

.. option:: --fields FIELDNAME1 [FIELDNAME2 ...], -f FIELDNAME1 [FIELDNAME2 ...]

    A list of fields to be displayed. Not valid with :option:`--template`.

.. option:: --options FIELD1.OPTION='VAL' [FIELD2.OPTION='VAL' ...], -o FIELD1.OPTION='VAL' [FIELD2.OPTION='VAL' ...]

    Set arbitrary options for discrete Fields using dot-attribute syntax.

.. option:: --refresh REFRESH, -r REFRESH

    The bar's refresh rate in seconds per cycle.

.. option:: --count TIMES, -n TIMES

    Print the bar this many times, then exit.

.. option:: --break-lines, -b

   Use a newline character at the end of every bar line.

.. option:: --icons FIELDNAME1='ICON1' [FIELDNAME2='ICON2' ...]

    A mapping of field names to icons.

.. option:: --unicode, -u

    Prefer Unicode versions of Field icons, if provided.

.. option:: --no-unicode, -U

    Prefer ASCII versions of Field icons, if provided.

.. option:: --config FILE, -c FILE

    The config file to use for default settings.

.. option:: --dump, -d

    Instead of running mybar, print a config file using options specified in the command.

.. option:: --debug

    Use debug mode. (Not implemented)

.. option:: --version, -v

    show program's version number and exit


Options for :option:`--fields`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  These options are not valid when using :option:`--template`:

.. option:: --separator 'FIELD_SEPARATOR', -s 'FIELD_SEPARATOR'

    The character used for joining fields. Only valid with :option:`--fields`.

.. option:: --join-empty, -j

    Include empty field contents instead of hiding them. Only valid with :option:`--fields`.


Examples
---------

Run mybar using your default config file:

.. code:: bash

    $ python -m mybar
    mymachine|Up 4d:14h:22m|CPU 05%|34C|Mem 8.6G|/:80.7G|Bat 100CHG|wifi|2023-11-10 16:34:18


Run mybar using custom icons:

.. code:: bash

    $ python -m mybar --icons uptime='⏱️' cpu_temp='🔥' mem_usage='🧠' battery='🔋'
    mymachine|⏱️4d:15h:7m|CPU 00%|🔥50C|🧠8.7G|/:80.7G|🔋100CHG|wifi|2023-11-10 17:19:20


Run mybar using specific fields:

.. code:: bash

    $ python -m mybar -f uptime cpu_usage mem_usage
    Up 4d:14h:25m|CPU 03%|Mem 8.6G


Run mybar with a custom field separator:

.. code:: bash

    $ python -m mybar -f uptime cpu_temp mem_usage -s ')('
    Up 4d:14h:27m)(CPU 01%)(Mem 8.5G


Run mybar using a custom format string:

.. code:: bash

    $ python -m mybar -t '{uptime}! [{cpu_usage}/{cpu_temp}]; {datetime}'
    Up 4d:14h:30m! [CPU 01%/36C]; 2023-11-10 16:42:11


Run mybar using more advanced custom field options:

.. code:: bash

    $ python -m mybar -o datetime.kwargs.fmt='%H:%M:%S.%f %m/%d/%Y'
    mymachine|Up 5d:23h:18m|CPU 05%|36C|Mem 9.7G|/:80.5G|Bat 97CHG|wifi|01:30:08.000588 11/12/2023


.. seealso::

   The :doc:`configuration` page shows more comprehensive customization options
   through the use of config files.

