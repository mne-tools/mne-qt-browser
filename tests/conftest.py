# License: BSD-3-Clause
# Copyright the MNE Qt Browser contributors.

import os
from pathlib import Path

import matplotlib
import mne
import pytest
from mne.conftest import _check_pyqtgraph
from mne.viz._figure import use_browser_backend
from pytest import StashKey
from qtpy.QtCore import QSettings
from refleak.testing import Snapshot, gc_collect_once

_store = {"Raw": {}, "Epochs_unicolor": {}, "Epochs_multicolor": {}}

# Stash each test's phase reports so fixtures can tell whether the test itself
# passed (see https://docs.pytest.org/en/stable/how-to/fixtures.html
# #using-markers-to-pass-data-to-fixtures).
_phase_report_key = StashKey()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Stash the status of each test phase."""
    outcome = yield
    rep = outcome.get_result()
    item.stash.setdefault(_phase_report_key, {})[rep.when] = rep


def _test_passed(request):
    """Return whether the test body (the "call" phase) passed."""
    if _phase_report_key not in request.node.stash:
        return True
    report = request.node.stash[_phase_report_key]
    return "call" in report and report["call"].outcome == "passed"


@pytest.fixture
def garbage_collect(request):
    """Garbage collect on exit."""
    yield
    gc_collect_once(request)


@pytest.fixture
def pg_backend(request, garbage_collect):
    """Use for pyqtgraph-specific test-functions.

    This overrides ``mne.conftest.pg_backend`` so that the leak check only runs
    when the test itself passed. When a test fails, pytest keeps its traceback
    (for reporting), which keeps that test's frame and hence its browser alive;
    checking for leaked browsers then would blame the failing test (or cascade
    into the next test using this fixture) for something the real failure caused.
    """
    _check_pyqtgraph(request)
    from mne_qt_browser._pg_figure import MNEQtBrowser

    with use_browser_backend("qt") as backend:
        backend._close_all()
        # Snapshot stores only ids, so it pins nothing alive; it lets us report
        # only browsers that this test itself leaked.
        snap = Snapshot(MNEQtBrowser, collect=False)
        yield backend
        backend._close_all()
        # This shouldn't be necessary, but let's make sure nothing is stale
        import mne_qt_browser

        mne_qt_browser._browser_instances.clear()
        if not _test_passed(request):
            return
        snap.assert_no_new(f"Closure of {request.node.name}", request=request)


@pytest.fixture(autouse=True, scope="session")
def _isolated_mne_config(tmp_path_factory):
    """Isolate the MNE config file for the test session.

    Closing a browser writes e.g. MNE_BROWSE_RAW_SIZE via mne.set_config, which
    would otherwise both trash the user's real config and make test behavior
    depend on whatever earlier runs left there (window sizes change pixel->data
    rounding in interaction tests).
    """
    config_dir = tmp_path_factory.mktemp("mne_config")
    mp = pytest.MonkeyPatch()
    mp.setenv("_MNE_FAKE_HOME_DIR", str(config_dir))
    yield
    mp.undo()


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
    # Browsers call mne.viz.backends._utils._qt_raise_window on show, which activates
    # and raises the window unless this is set
    matplotlib.rcParams["figure.raise_window"] = False
    warning_lines = r"""
    error::
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
