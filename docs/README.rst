######
mybar
######
 
*Craft highly customizable status bars with ease.*


About mybar
======
**mybar** is a code library and command line tool written in Python for making
status bars.

It aims to help users create custom status bars with minimal effort, intuitive
controls and every aspect of a bar available to be customized.

::

   $ python -m mybar --template '{uptime} [{cpu_usage}/{cpu_temp}] | {battery}'
   Up 4d:14h:19m [CPU 03%/36C] | Bat 99CHG


Install mybar
========

**mybar** supports Python 3.11+.

It can be installed from the `Python Package Index`::

   $ python -m pip install mybar


Use mybar
======

By default, **mybar** looks at config files to load its options.

Running the **mybar** command line utility using your default config file is as simple as::

   $ python -m mybar

The first time you run **mybar**, it will check if you have a config file in the default location.
In the following example, my home directory is ``'/home/sam'``::

   -- mybar --
   The default config file at '/home/sam/.config/mybar/conf.json' does not exist.
   Would you like to make it now? [Y/n] y
   Wrote new config file to '/home/sam/.config/mybar/conf.json'

You can also skip writing a config file and **mybar** will start running a bar with default
parameters::

   Would you like to make it now? [Y/n] n
   strangelove|Up 4d:14h:54m|CPU 04%|36C|Mem 3.8G|/:50.8G|Bat 100CHG|wifi|2023-06-01 17:06:04

See the `manual` for details on the command line arguments **mybar** accepts.


Concepts
=========

This section introduces the core concepts that make customizing **mybar** possbile.

- Bar
      The status bar
- Field
      A part of the `Bar` containing information, sometimes called a "module" by
      other status bar frameworks
- field function
      The function a `Field` runs to determine what it should contain
- refresh cycle
      The time it takes the `Bar` to run all its fields and update its contents once
- refresh rate
      How often the `Bar` updates what it says, in seconds per refresh
- interval
      How often a `Field` runs its field function, in seconds per cycle
- separator
      A string that separates one `Field` from another
- format string
      A special string that controls how `Fields` and their contents are displayed
- icon
      A string appearing with each `Field`, usually unique to each


To customize **mybar** to your liking without using the `Python API`, you can use
`config files`
or `command line arguments`.


Configure
==========


Examples
=========


Advanced Usage // Field Funcs
============

`Field funcs` are Python functions that return the contents of a `Field`.
Below is documentation for each function found in `mybar.field_funcs`.

