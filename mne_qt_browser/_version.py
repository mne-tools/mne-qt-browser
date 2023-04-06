"""The version number."""

try:
    from importlib.metadata import version
    __version__ = version("mne_qt_browser")
except Exception:
    try:
        from ._version import __version__
    except ImportError:
        __version__ = '0.0.0'
