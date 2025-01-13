# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Martin Schulz <dev@earthman-music.de>
#
# License: BSD-3-Clause

import os

import pytest
from mne.conftest import garbage_collect, pg_backend, raw_orig  # noqa: F401

_store = {"Raw": {}, "Epochs_unicolor": {}, "Epochs_multicolor": {}}


def pytest_configure(config):
    """Configure pytest options."""
    # Markers
    for marker in ("benchmark", "pgtest", "slowtest"):
        config.addinivalue_line("markers", marker)
    if "_MNE_BROWSER_BACK" not in os.environ:
        os.environ["_MNE_BROWSER_BACK"] = "true"
    warning_lines = r"""
    error::
    # PySide6
    ignore:Enum value .* is marked as deprecated:DeprecationWarning
    ignore:Function.*is marked as deprecated, please check the documentation.*:DeprecationWarning
    ignore:Failed to disconnect.*:RuntimeWarning
    """  # noqa: E501
    for warning_line in warning_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)


@pytest.fixture(scope="session")
def store():
    """Yield our storage object."""
    yield _store


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print our benchmark results (if present)."""
    if not any(len(_store[key]) for key in _store):
        return
    writer = terminalreporter
    writer.line("")  # newline
    writer.write_sep("=", "benchmark results")
    for type_name, results in _store.items():
        writer.write_sep("-", type_name)
        for name, vals in results.items():
            writer.line(
                f"{name}:\n"
                f"    Horizontal: {vals['h']:6.2f}\n"
                f"    Vertical:   {vals['v']:6.2f}"
            )
