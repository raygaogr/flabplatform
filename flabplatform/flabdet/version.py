from flabplatform import parse_version_info

__version__ = '0.0.1rc1'

version_info = parse_version_info(__version__)

__all__ = ['__version__', 'version_info']
