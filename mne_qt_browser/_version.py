"""The version number."""

try:
    from importlib.metadata import version
    __version__ = version("mne_qt_browser")
except Exception:
    try:
        from .__version import __version__  # written by setuptools_scm
    except ImportError:
        __version__ = '0.0.0'
