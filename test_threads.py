import threading
from time import sleep
from typing import Callable

class Field(threading.Thread):
    def __init__(self, name, func, interval, sentinel: bool = True, *args, **kwargs):
        self.func = func
        self.interval = interval
        self.SENTINEL = sentinel
        self.args = args
        self.kwargs = kwargs
        looped = self._loop # (func, args, kwargs)
        super().__init__(target=looped, name=func.__name__ if name is None else name)

    def _loop(self):
        while self.SENTINEL:
            self.func(*self.args, **self.kwargs)
            sleep(self.interval)
        
        # self.join()

    def run(self, ):
        try:
            self._loop()
        except KeyboardInterrupt as e:
            self.terminate()
        except Exception as e:
            self.on_exception(e)
        # finally:
            # self.join()
        # self.join()


    def on_exception(self, exc):
        print(f"Field {self.name!r} got exception: {exc!r}")
        self.SENTINEL = False
        self.join()

    def terminate(self):
        print("Got KeyboardInterrupt")
        self.SENTINEL = False
        self.join()


class Bar:

    _field_funcs = {
        'hostname': None, # get_hostname,
    }

    def __init__(self, fields=None, refresh_rate=0.2, **kwargs):
        if isinstance(fields, list) or (names := kwargs.get('field_names')):
            fields = {name: self._field_funcs.get(name) for name in names}


def get_stuff():
    print("Works")

