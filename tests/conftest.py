# License: BSD-3-Clause
# Copyright the MNE Qt Browser contributors.

import os
import re
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


# Unset means "allow every skip". Set it to e.g. "^$" to turn all skips into errors,
# or to a pattern matching the reasons that are expected on this configuration.
MNE_TEST_ALLOW_SKIP = os.getenv("MNE_TEST_ALLOW_SKIP", None)
_valid_skips_re = re.compile(MNE_TEST_ALLOW_SKIP or ".*", re.DOTALL)


def pytest_report_header(config):
    """Add the allowed skips to the pytest run header."""
    if MNE_TEST_ALLOW_SKIP is None:
        return []
    return [f"Allowed skips: {MNE_TEST_ALLOW_SKIP!r}"]


def pytest_report_teststatus(report, config):
    """Turn unexpected skips into errors (adapted from mne-python's conftest)."""
    # Both report types matter: CollectReport covers skipif marks, TestReport covers
    # skips raised from a test body (e.g. importorskip)
    if MNE_TEST_ALLOW_SKIP is None or report.outcome != "skipped":
        return
    if isinstance(report.longrepr, tuple):
        file, lineno, reason = report.longrepr
    else:
        file, lineno, reason = "<unknown>", 1, str(report.longrepr)
    if _valid_skips_re.match(reason):
        return
    # xfails are not skips, but are reported as such (by mark, or by traceback)
    if (
        getattr(report, "keywords", {}).get("xfail", False)
        or " pytest.xfail( " in reason
    ):
        return
    reason = reason.removeprefix("Skipped: ")
    report.longrepr = f"{file}:{lineno}: UNEXPECTED SKIP: {reason!r}"
    report.outcome = "error" if isinstance(report, pytest.TestReport) else "failed"
    return report.outcome, report.outcome[0].upper(), "UNEXPECTED SKIP"


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


@pytest.fixture(autouse=True)
def _isolated_mne_config(tmp_path):
    """Isolate the MNE config file, one per test.

    Closing a browser writes e.g. MNE_BROWSE_RAW_SIZE via mne.set_config, which
    would otherwise both trash the user's real config and make test behavior
    depend on whatever earlier runs left there. A directory per test (rather
    than per session) also stops a test that resizes its window from changing
    the size every later browser opens at, since window size feeds the
    pixel->data rounding that interaction tests depend on.
    """
    config_dir = tmp_path / "mne_config"
    config_dir.mkdir()
    mp = pytest.MonkeyPatch()
    mp.setenv("_MNE_FAKE_HOME_DIR", str(config_dir))
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path):
    """Isolate QSettings to a temporary file, one per test.

    Closing a browser writes every entry of ``qsettings_params`` back out, so a
    file shared across tests would let one test's settings (e.g. the
    ``downsampling=17`` that ``test_pg_settings_dialog`` sets) silently change
    how every later browser renders. A fresh file per test means each one starts
    from the documented defaults.
    """
    ini_path = tmp_path / "mne-qt-browser-test.ini"

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
