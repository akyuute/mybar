Config Files
=============

**mybar** uses the `Scuff <https://github.com/akyuute/scuff>`_
language to process data from and write to config files.

The default config file location is ``~/.config/mybar/mybar.conf``



Configuring **mybar**
----------------------


_`Bar options`
~~~~~~~~~~~~~~
    The following options are used to control how the bar runs:

    - `refresh`
        `(float)` The bar's refresh rate in seconds per cycle.

    - `separator`
        `(string)` A string or list of strings (one for ASCII terminals, one
        for Unicode terminals) used to separate each Field.

    - `count`
        `(integer)` Print the bar this many times before the program quits.

    - `template`
        `(string)` A template string to use instead of `field_order`.

    - `break_lines`
        `(bool)` Write each Bar on a new line every refresh cycle.

    - `unicode`
        `(bool)` Use Unicode Field icons and separator, if given.

    - `field_order`
        `(list)` A list of Fields to display if `template` is unset.
        For example:

        .. code:: py

            field_order [uptime cpu_usage cpu_temp net_stats datetime]

    - `field_icons`
        `(mapping)` A mapping of Field names to icons or lists of icons
        (one for ASCII terminals, one for Unicode terminals) for each Field.
        For example:

        .. code:: py

            field_icons {
                uptime "Up "
                cpu_usage ["CPU ", "📈"]
                cpu_temp ["", "\uf06d "]
            }

    - `clock_align`
        `(bool)` Print the bar at the top of each second.

    - `join_empty_fields`
        `(bool)` Show separators around Fields, even when they are empty.

    - `thread_cooldown`
        `(float)` How long a field thread loop sleeps after checking if
        the bar is still running.
        Between executions, unlike async fields, a threaded field sleeps
        for several iterations of `thread_cooldown` seconds that always
        add up to :attr:`Field.interval` seconds.
        Between sleeps, it checks if the bar has stopped.
        A shorter cooldown means more chances to check if the bar has
        stopped and a faster exit time.

    - Field definitions
        Field definitions are mappings with Field options used to override
        defaults. See `Field options`_ for a complete reference. You may use the
        rest of the file to customize specific Fields in the `field_order` list.
        For example:

        .. code:: py

            datetime {
                interval 3
                fmt "{} o'clock"
            }

            cpu_usage {threaded=False}

            
.. note::
    See :doc:`fields` for a description of default fields.



_`Field options`
~~~~~~~~~~~~~~~~~

    - `icon`
        `(string)` Positioned in front of Field contents or in place of ``{icon}`` in `template`, if provided

    - `template`
        `(string)` A curly-brace format string.
        This parameter is **required** if `icon` is ``None``.

        Valid placeholders:
            - ``{icon}`` references `icon`
            - ``{}`` references Field contents

        Example:
            | When the Field's current contents are ``'69F'`` and its icon is ``'TEMP'``,
            | ``template='[{icon}]: {}'`` shows as ``'[TEMP]: 69F'``

    - `interval`
        `(float)` How often in seconds per update Field contents are updated, defaults to ``1.0``

    - `clock_align`
        `(bool)` Update contents at the start of each second, defaults to ``False``

    - `timely`
        `(bool)` Run the Field function as soon as possible after every refresh,
        defaults to ``False``

    - `overrides_refresh`
        `(bool)` Ensure updates to this Field re-print the Bar between refreshes,
        defaults to ``False``

    - `threaded`
        `(bool)` Run this Field in a separate thread, defaults to ``False``

    - `always_show_icon`
        `(bool)` Show icons even when contents are empty, defaults to ``False``

    - `run_once`
        `(bool)` Permanently set contents by running the `func` only once, defaults to ``False``

    - `constant_output`
        `(string)` Permanently set contents instead of running a function

    - `args`
        `(list)` Positional args passed to `func`

    - `kwargs`
        `(mapping)` Keyword args passed to `func`


- Custom Fields
    New positionable Fields with custom values are specified with the `custom` option.
    For example::

        my_custom_field {
            custom true
            constant_value "Hello!"
        }



Here is an example config file:

.. code:: py

    refresh 0.5
    separator ["|", "∦"]
    unicode yes

    field_order [
        uptime
        my_custom_field
        cpu_usage
        cpu_temp
        mem_usage
        # disk_usage
        battery
        net_stats
        datetime
    ]

    field_icons {
        # Unicode codepoints in the second slot are for Fontawesome icons
        uptime ["Up ", "\uf2f2 "]
        cpu_usage ["CPU ", "\uf3fd "]
        cpu_temp ["", "\uf06d "]
        mem_usage ["MEM ", "\uf2db "]
        battery "BAT "
        net_stats ["", "\uf1eb "]
    }

    datetime {interval 10}

    # Give the time function a different format:
    datetime.kwargs.fmt '%H:%M:%S.%f'

    my_custom_field {
        custom yes
        constant_output "Hi!"
        template " {} "
    }

