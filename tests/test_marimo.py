# License: BSD-3-Clause
# Copyright the MNE Qt Browser contributors.

"""Test the marimo event pump with a fake marimo module (no marimo needed)."""

import asyncio
import sys
import types

import pytest
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QLabel

from mne_qt_browser import _pg_figure


@pytest.fixture
def in_marimo(monkeypatch):
    """Make _setup_marimo believe it is running inside a marimo notebook."""
    fake = types.ModuleType("marimo")
    fake.running_in_notebook = lambda: True
    monkeypatch.setitem(sys.modules, "marimo", fake)
    monkeypatch.setattr(_pg_figure, "_MARIMO_PUMP", None)
    return fake


def test_marimo_pump(qapp, in_marimo, monkeypatch):
    """Test the pump lifecycle: start, service Qt, drain on close, stop, restart."""
    monkeypatch.setattr(_pg_figure, "_MARIMO_PUMP_GRACE", 0.5)
    monkeypatch.setattr(_pg_figure, "_MARIMO_PUMP_DRAIN", 0.3)
    qapp.closeAllWindows()  # leftover windows would keep the pump alive
    assert _pg_figure._setup_marimo() is None  # no running asyncio loop -> no pump
    widget = QLabel("marimo pump test")

    async def pump_and_close():
        task = _pg_figure._setup_marimo()
        assert task is not None and not task.done()
        assert _pg_figure._setup_marimo() is task  # no second pump while one runs

        widget.show()
        ticks = [0]
        probe = QTimer()
        probe.timeout.connect(lambda: ticks.__setitem__(0, ticks[0] + 1))
        probe.start(10)
        # While this sleep blocks, only the pump can service Qt's event queue
        await asyncio.sleep(0.5)
        probe.stop()
        assert not task.done()
        assert ticks[0] > 5, f"window frozen, only {ticks[0]} ticks"

        doomed = QLabel("deleted on close")  # closing a window schedules deletions
        doomed.deleteLater()
        widget.close()
        await asyncio.sleep(0.1)  # still within the drain
        assert not task.done(), "pump stopped before the close finished"
        with pytest.raises(RuntimeError, match="deleted"):  # wording differs by binding
            doomed.objectName()  # the pump flushed the deferred delete (gh-449)
        await asyncio.wait_for(task, timeout=5)  # and then stops

    asyncio.run(pump_and_close())

    # A pump left pending by a kernel that went away must not be reused
    widget.show()

    async def start_only():
        return _pg_figure._setup_marimo()

    stale = asyncio.run(start_only())  # left pending when that loop closes

    async def restart_then_give_up():
        task = _pg_figure._setup_marimo()
        assert task is not stale
        await asyncio.sleep(0.1)  # let the new pump see the window before it closes
        widget.close()
        await asyncio.wait_for(task, timeout=5)
        # nothing on screen now, so the next pump gives up instead of spinning forever
        await asyncio.wait_for(_pg_figure._setup_marimo(), timeout=5)

    asyncio.run(restart_then_give_up())


def test_marimo_no_pump(qapp, in_marimo, monkeypatch):
    """Test that no pump starts when something else is already driving Qt."""

    async def setup():
        return _pg_figure._setup_marimo()

    in_marimo.running_in_notebook = lambda: False
    assert asyncio.run(setup()) is None  # marimo imported, but not the notebook kernel

    in_marimo.running_in_notebook = lambda: True

    class FakeQasyncLoop(asyncio.SelectorEventLoop):
        pass

    FakeQasyncLoop.__module__ = "qasync"  # a loop that drives Qt itself

    loop = FakeQasyncLoop()
    try:
        assert loop.run_until_complete(setup()) is None
    finally:
        loop.close()
