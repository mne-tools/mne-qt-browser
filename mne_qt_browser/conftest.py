# -*- coding: utf-8 -*-
# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Martin Schulz <dev@earthman-music.de>
#
# License: BSD-3-Clause


import pytest

from mne.conftest import (raw_orig, pg_backend, garbage_collect)  # noqa: F401

_store = dict()


def pytest_configure(config):
    """Configure pytest options."""
    # Markers
    for marker in ('benchmark',):
        config.addinivalue_line('markers', marker)


@pytest.fixture(scope='session')
def store():
    """Yield our storage object."""
    yield _store


def pytest_sessionfinish(session, exitstatus):
    """Print our benchmark results (if present)."""
    if len(_store):
        from py.io import TerminalWriter
        writer = TerminalWriter()
        writer.line()  # newline
        writer.sep('=', 'benchmark results')
        for name, vals in _store.items():
            writer.line(
                f'{name}:\n'
                f'    Horizontal: {vals["h"]:6.2f}\n'
                f'    Vertical:   {vals["v"]:6.2f}')
