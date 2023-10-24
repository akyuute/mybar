Config Files
=============

Here is the config file syntax.


Assignments
------------
Values can be assigned to variables with or without an equals sign (=)::
    
    my_favorite_number = 42
    my_favorite_color "Magenta"

Variables can be left empty as long as they have an equals sign::

    best_food "Pizza"
    you_disagree =   # Evaluated as ``None``


Data Types
-----------

- Numbers
    Numbers can be positive integers or floats::

        1 1.2 1_000 0.123 .123

- Booleans
    The boolean values ``True`` and ``False`` are given using these variants::

        True true yes
        False false no

- Strings
    Single-line strings can be enclosed by single quotes ('), double quotes (")
    or backticks (`), and multiline strings are enclosed by three of any of those::

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
    Lists are enclosed by square brackets ([]). Elements inside lists are separated by spaces, commas or line breaks::

        field_order [
            uptime,
            cpu_usage cpu_temp
            datetime
        ]


- Objects
    Objects, which are groups of key-value pairs, are enclosed by curly braces
    ({}). Key names must be valid variable names, meaning they have no spaces and
    don't contain symbols except underscore (_).
    Values may be any expression, even another object::

        good_movies {
            Sharknado yes
            Jaws = false
            Sharknado2 {
                just_for_example =
                here_is_a_nested_object 3.14
            }
        }

