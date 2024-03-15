# -*- coding: utf-8 -*-
"""Backports for older installations."""

# Author: Martin Schulz <dev@earthman-music.de>
#
# License: BSD-3-Clause

import platform
import sys
from contextlib import contextmanager

from mne.utils import logger
from mne.viz.backends._utils import _init_mne_qtapp, _qt_raise_window

# pytestqt
try:
    from pytestqt.exceptions import capture_exceptions
except ImportError:
    logger.debug(
        "If pytest-qt is not installed, errors from inside the event loop will be occluded "
        "and it will be harder to trace back the cause."
    )

    @contextmanager
    def capture_exceptions():
        yield []
