Config Files
=============

Config files are written as a series of variable assignments with a target
and a value.

Assigning Variables
--------------------
Values can be assigned to variables with or without an equals sign (``=``)::

    my_favorite_number = 42
    my_favorite_color "Magenta"
    is_but_a_flesh_wound= yes

Variables will not override default options if they are left empty with a
trailing equals sign::

    run_once = yes  # Overrides ``no``
    count =  # Defaults to 1

More complex options represented by nested dicts in Python may be
specified using object attributes::

    uptime.kwargs.dynamic = no  # Display uptime without special formatting.

Learn more about Field options in particular: `Field options`_


Data Types
-----------

- Numbers
    Numbers can be positive integers or floats::

        1 1.2 1_000 0.123 .123_4

- Booleans
    The boolean values ``True`` and ``False`` are given using these variants::

        True true yes
        False false no

- Strings
    Single-line strings can be enclosed by single quotes (``'``), double
    quotes (``"``) or backticks (`````), and multiline strings are enclosed by
    three of any of those::

        foo "abc"
        bar 'def'
        baz '''Hi,
                did you know
                    you're cute?
                        '''

    Strings placed right next to each other are concatenated::
        
        first = "ABC"
        second = "DEF"
        first_plus_second = "ABC"  "DEF"
        concatenated = "ABCDEF"
                    
- Lists
    Lists are enclosed by square brackets (``[]``).
    Elements inside lists are separated by spaces, commas or line breaks::

        groceries [
            bread,
            milk eggs
            bacon
        ]

- Objects
    Objects, which are groups of key-value pairs, are enclosed by curly braces
    (``{}``). Key names must be valid variable names, meaning they have no
    spaces and don't contain symbols except underscore (``_``).
    Values may be any expression, even another object::

        good_movies {
            Sharknado yes
            Jaws = false
            Sharknado2 {
                just_for_example '...'
                here_is_a_nested_object 3.14
            }
        }


- Comments
    Single-line comments are made using the ``#`` symbol::

        option = "The parser reads this."
        # But this is a comment.
            #And so is this.
        option2 = "# But not this; It's inside a string."
        # The parser ignores everything between ``#`` and the end of the line.
         #   ignore = "Comment out any lines of code you want to skip."



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

    - `unicode`
        `(bool)` Use Unicode Field icons and separator, if given.

    - `field_order`
        `(list)` A list of Fields to display if `template` is unset.
        For example::

            field_order [uptime cpu_usage cpu_temp net_stats datetime]

    - `field_icons`
        `(object)` An object mapping Field names to icons or lists of icons
        (one for ASCII terminals, one for Unicode terminals) for each Field.
        For example::

            field_icons {
                uptime "Up "
                cpu_usage ["CPU ", "📈"]
                cpu_temp ["", "\uf06d "]
            }


    - Field definitions
        Field definitions are objects with Field options used to override
        defaults. See `Field options`_ for a complete reference. You may use the
        rest of the file to customize specific Fields in the `field_order` list.
        For example::

            datetime {
                interval 3
                fmt "{} o'clock"
            }

            cpu_usage {threaded=False}

            

_`Field options`
~~~~~~~~~~~~~~~~~

  ..
        name: FieldName = None,
        func: Callable[P, str] = None,
        icon: str = '',
        template: FormatStr = None,
        interval: float = 1.0,
        clock_align: bool = False,
        timely: bool = False,
        overrides_refresh: bool = False,
        threaded: bool = False,
        always_show_icon: bool = False,
        run_once: bool = False,
        constant_output: str = None,
        bar: Bar_T = None,
        args: Args = None,
        kwargs: Kwargs = None,
        setup: Callable[P, P.kwargs] = None,


- Custom Fields
    Positionable Fields with custom values are specified with the `custom` option.
    For example::

        my_custom_field = {
            custom true
            constant_value "Hello!"
        }



Here is an example config file::

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

