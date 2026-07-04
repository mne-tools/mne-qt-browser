# License: BSD-3-Clause
# Copyright the MNE Qt Browser contributors.

import os
from pathlib import Path

import mne
import pytest
from mne.conftest import garbage_collect, pg_backend  # noqa: F401
from qtpy.QtCore import QSettings

_store = {"Raw": {}, "Epochs_unicolor": {}, "Epochs_multicolor": {}}


@pytest.fixture(autouse=True, scope="session")
def _isolated_qsettings(tmp_path_factory):
    """Isolate QSettings to a temporary file for the test session."""
    ini_path = tmp_path_factory.mktemp("qsettings") / "mne-qt-browser-test.ini"

    def _fake_qsettings(*_args, **_kwargs):
        return QSettings(str(ini_path), QSettings.IniFormat)

    mp = pytest.MonkeyPatch()
    mp.setattr("mne_qt_browser._pg_figure.QSettings", _fake_qsettings)
    yield
    mp.undo()


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
    ignore:Function.*is marked as deprecated, please check the .*:DeprecationWarning
    ignore:Failed to disconnect.*:RuntimeWarning
    """
    for warning_line in warning_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)


@pytest.fixture(scope="session")
def store():
    """Yield our storage object."""
    yield _store


@pytest.fixture(scope="session")
def raw_orig():
    """Raw instance loaded from local test_raw.fif."""
    raw_path = Path(__file__).parent / "test_raw.fif"
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="ERROR")
    return raw


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
