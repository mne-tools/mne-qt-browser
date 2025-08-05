try:
    from importlib.metadata import version

    __version__ = version("mne_qt_browser")
except Exception:
    __version__ = "0.0.0"

# Keep references to all created brower instances to prevent them from being
# garbage-collected prematurely
_browser_instances = list()
