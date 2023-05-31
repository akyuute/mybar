__all__ = (
    'logger'
)


import logging
import os.path

from .constants import CONFIG_FILE


logging.basicConfig(
    level='DEBUG',
    filename=os.path.dirname(CONFIG_FILE) + '/mybar.log',
    filemode='w',
    datefmt='%Y-%m-%d_%H:%M:%S.%f',
    format='[{asctime}] ({levelname}:{name}) {message}',
    style='{',
)
logger = logging.getLogger(__name__)

