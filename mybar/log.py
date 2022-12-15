__all__ = (
    'logger'
)


logging.basicConfig(
    level='DEBUG',
    filename=os.path.expanduser('~/.mybar.log'),
    filemode='w',
    datefmt='%Y-%m-%d_%H:%M:%S.%f',
    format='[{asctime}] ({levelname}:{name}) {message}',
    style='{',
)
logger = logging.getLogger(__name__)

