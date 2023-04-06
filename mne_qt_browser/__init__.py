try:
    from importlib.metadata import version
    __version__ = version("mne_qt_browser")
except Exception:
    try:
        from ._version import __version__  # written by setuptools_scm
    except ImportError:
        __version__ = '0.0.0'

# All created brower-instances are listed here for a reference to avoid having
# them garbage-collected prematurely.
_browser_instances = list()
