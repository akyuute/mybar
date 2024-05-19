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
                   [--from-icons FIELDNAME1='ICON1' [FIELDNAME2='ICON2' ...]]
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

.. option:: --icons FIELDNAME1='ICON1' [FIELDNAME2='ICON2' ...], -i FIELDNAME1='ICON1' [FIELDNAME2='ICON2' ...]

   A mapping of field names to icons.

.. option:: --from-icons FIELDNAME1='ICON1' [FIELDNAME2='ICON2' ...]

    A mapping of Field names to icons.
    Use for the Field order as well.

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


``--fields/-f`` Specify which fields to show:

.. code:: bash

   $ python -m mybar -f hostname disk_usage cpu_temp datetime
   mymachine|/:88.3G|43C|2023-08-01 23:18:22


``--icons/-i`` Set new icons for each field:

.. code:: bash

   $ python -m mybar -i uptime='⏱️' cpu_temp='🔥' mem_usage='🧠' battery='🔋'
   mymachine|⏱️4d:15h:7m|CPU 00%|🔥50C|🧠8.7G|/:80.7G|🔋100CHG|wifi|2023-11-10 17:19:20


``--template/-t`` Use a custom format template:

.. code:: bash

   $ python -m mybar -t '@{hostname}: ({uptime} | {cpu_usage}, {cpu_temp})  [{datetime}]'
   @mymachine: (Up 1d:12h:17m | CPU 02%, 44C)  [2023-08-01 23:31:26]


``--separator/-s`` Change the field separator:

.. code:: bash

   $ python -m mybar -f hostname uptime cpu_usage -s ' ][ '
   mymachine ][ Up 1d:12h:11m ][ CPU 00%


``--count/-n`` Run the bar a specific number of times:

.. code:: bash

   $ python -m mybar -f hostname cpu_usage datetime -n 3 --break-lines
   mymachine|CPU 00%|2023-08-01 23:40:26
   mymachine|CPU 00%|2023-08-01 23:40:27
   mymachine|CPU 00%|2023-08-01 23:40:28
   $


``--refresh/-r`` Set the bar's refresh rate:

.. code:: bash

   $ python -m mybar -f hostname cpu_usage datetime -n 3 -r 10 --break-lines
   mymachine|CPU 00%|2023-11-24 04:25:31
   mymachine|CPU 00%|2023-11-24 04:25:41
   mymachine|CPU 00%|2023-11-24 04:25:51
   $


``--options/-o`` Set arbitrary options for the bar or any field:

.. code:: bash

   $ python -m mybar -t 'Time: {datetime}' -o datetime.kwargs.fmt='%H:%M:%S.%f'
   Time: 01:19:55.000229


``--config/-c`` Use a specific config file:

.. code:: bash

   $ python -m mybar -c ~/.config/mybar/my_other_config_file.conf


.. seealso::

   The :doc:`configuration` page shows more comprehensive customization options
   through the use of config files.

