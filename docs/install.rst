Installation
=============

To install **mybar**, open your terminal.
Use `pip <https://docs.python.org/3/installing/index.html>`_ to install the package::

    $ python -m pip install mybar

Then run it with Python::

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

