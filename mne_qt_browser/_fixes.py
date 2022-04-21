# -*- coding: utf-8 -*-
"""Backports for MNE versions (mostly)."""

# Author: Martin Schulz <dev@earthman-music.de>
#
# License: BSD-3-Clause

from contextlib import contextmanager
import platform
import sys

from mne.utils import logger

###############################################################################
# MNE 1.0+
try:
    from mne.viz.backends._utils import _qt_raise_window
except ImportError:
    def _qt_raise_window(win):
        try:
            from matplotlib import rcParams
            raise_window = rcParams['figure.raise_window']
        except ImportError:
            raise_window = True
        if raise_window:
            win.activateWindow()
            win.raise_()

try:
    from mne.viz.backends._utils import _init_mne_qtapp
except ImportError:
    from mne.viz.backends._utils import _init_qt_resources

    def _init_mne_qtapp(enable_icon=True, pg_app=False):
        """Get QApplication-instance for MNE-Python.

        Parameter
        ---------
        enable_icon: bool
            If to set an MNE-icon for the app.
        pg_app: bool
            If to create the QApplication with pyqtgraph. For an until know
            undiscovered reason the pyqtgraph-browser won't show without
            mkQApp from pyqtgraph.

        Returns
        -------
        app: ``qtpy.QtWidgets.QApplication``
            Instance of QApplication.
        """
        from qtpy.QtWidgets import QApplication
        from qtpy.QtGui import QIcon

        app_name = 'MNE-Python'
        organization_name = 'MNE'

        # Fix from cbrnr/mnelab for app name in menu bar
        if sys.platform.startswith("darwin"):
            try:
                # set bundle name on macOS (app name shown in the menu bar)
                from Foundation import NSBundle
                bundle = NSBundle.mainBundle()
                info = (bundle.localizedInfoDictionary()
                        or bundle.infoDictionary())
                info["CFBundleName"] = app_name
            except ModuleNotFoundError:
                pass

        if pg_app:
            from pyqtgraph import mkQApp
            app = mkQApp(app_name)
        else:
            app = (QApplication.instance()
                   or QApplication(sys.argv or [app_name]))
            app.setApplicationName(app_name)
        app.setOrganizationName(organization_name)

        if enable_icon:
            # Set icon
            _init_qt_resources()
            kind = 'bigsur-' if platform.mac_ver()[0] >= '10.16' else ''
            app.setWindowIcon(QIcon(f":/mne-{kind}icon.png"))

        return app

###############################################################################
# pytestqt
try:
    from pytestqt.exceptions import capture_exceptions
except ImportError:
    logger.debug('If pytest-qt is not installed, the errors from inside '
                 'the Qt-loop will be occluded and it will be harder '
                 'to trace back the cause.')

    @contextmanager
    def capture_exceptions():
        yield []
