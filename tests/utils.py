import types

import numpy as np
from pyqtgraph import Point
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QGraphicsView

from mne_qt_browser._fixes import capture_exceptions


def add_test_browser_methods(fig):
    """Attach test helper methods to an MNEQtBrowser instance."""

    def _fake_keypress(self, key, fig2=None):
        fig2 = fig2 or self
        if key.isupper():
            key = key.lower()
            modifier = Qt.ShiftModifier
        elif key.startswith("shift+"):
            key = key[6:]
            modifier = Qt.ShiftModifier
        else:
            modifier = Qt.NoModifier
        with capture_exceptions() as exceptions:
            QTest.keyPress(fig2, self.mne.keyboard_shortcuts[key]["qt_key"], modifier)
        for exc in exceptions:
            raise RuntimeError(
                "There as been an "
                f"{exc[0]} inside the Qt event loop (look above for traceback)."
            )

    def _fake_click(
        self,
        point,
        add_points=None,
        fig2=None,
        ax=None,
        xform="ax",
        button=1,
        kind="press",
        modifier=None,
    ):
        add_points = add_points or list()
        QTest.qWaitForWindowExposed(self)
        QTest.qWait(10)
        if button == 1:
            button_qt = Qt.LeftButton
        else:
            button_qt = Qt.RightButton
        fig2 = ax or fig2 or self.mne.view
        if xform == "ax":
            view_width = fig2.width()
            view_height = fig2.height()
            x = view_width * point[0]
            y = view_height * (1 - point[1])
            point_qt = Point(x, y)
            for idx, apoint in enumerate(add_points):
                x2 = view_width * apoint[0]
                y2 = view_height * (1 - apoint[1])
                add_points[idx] = Point(x2, y2)
        elif xform == "data":
            fig2 = self.mne.view
            point_qt = self.mne.viewbox.mapViewToScene(Point(*point))
            for idx, apoint in enumerate(add_points):
                add_points[idx] = self.mne.viewbox.mapViewToScene(Point(*apoint))
        elif xform == "none" or xform is None:
            if isinstance(point, tuple | list):
                point_qt = Point(*point)
            else:
                point_qt = Point(point)
            for idx, apoint in enumerate(add_points):
                if isinstance(apoint, tuple | list):
                    add_points[idx] = Point(*apoint)
                else:
                    add_points[idx] = Point(apoint)
        else:  # fallback
            point_qt = Point(*point)

        with capture_exceptions() as exceptions:
            widget = fig2.viewport() if isinstance(fig2, QGraphicsView) else fig2
            if kind == "press":
                _mouseClick(
                    widget=widget, pos=point_qt, button=button_qt, modifier=modifier
                )
            elif kind == "release":
                _mouseRelease(
                    widget=widget, pos=point_qt, button=button_qt, modifier=modifier
                )
            elif kind == "motion":
                _mouseMove(
                    widget=widget, pos=point_qt, buttons=button_qt, modifier=modifier
                )
            elif kind == "drag":
                _mouseDrag(
                    widget=widget,
                    positions=[point_qt] + add_points,
                    button=button_qt,
                    modifier=modifier,
                )
        for exc in exceptions:
            raise RuntimeError(
                "There as been an "
                f"{exc[0]} inside the Qt event loop (look above for traceback)."
            )
        QTest.qWait(50)

    def _fake_scroll(self, x, y, step, fig2=None):  # noqa: ARG001 (API compatibility)
        self.vscroll(step)

    def _click_ch_name(self, ch_index, button):
        self.mne.channel_axis.repaint()
        QTest.qWait(100)
        if not self.mne.butterfly:
            ch_name = str(self.mne.ch_names[self.mne.picks[ch_index]])
            xrange, yrange = self.mne.channel_axis.ch_texts[ch_name]
            x = np.mean(xrange)
            y = np.mean(yrange)
            self._fake_click((x, y), fig2=self.mne.view, button=button, xform="none")

    def _fake_click_on_toolbar_action(self, action_name, wait_after=500):
        for action in self.mne.toolbar.actions():
            if not action.isSeparator():
                if action.iconText() == action_name:
                    action.trigger()
                    break
        else:  # no break
            raise ValueError(f"action_name={action_name!r} not found")
        QTest.qWait(wait_after)

    for func in (
        _fake_keypress,
        _fake_click,
        _fake_scroll,
        _click_ch_name,
        _fake_click_on_toolbar_action,
    ):
        setattr(fig, func.__name__, types.MethodType(func, fig))
    return fig


# mouse testing functions adapted from pyqtgraph (pyqtgraph.tests.ui_testing.py)
def _mousePress(widget, pos, button, modifier=None):
    if modifier is None:
        modifier = Qt.KeyboardModifier.NoModifier
    event = QMouseEvent(
        QEvent.Type.MouseButtonPress, pos, button, Qt.MouseButton.NoButton, modifier
    )
    QApplication.sendEvent(widget, event)


def _mouseRelease(widget, pos, button, modifier=None):
    if modifier is None:
        modifier = Qt.KeyboardModifier.NoModifier
    event = QMouseEvent(
        QEvent.Type.MouseButtonRelease, pos, button, Qt.MouseButton.NoButton, modifier
    )
    QApplication.sendEvent(widget, event)


def _mouseMove(widget, pos, buttons=None, modifier=None):
    if buttons is None:
        buttons = Qt.MouseButton.NoButton
    if modifier is None:
        modifier = Qt.KeyboardModifier.NoModifier
    event = QMouseEvent(
        QEvent.Type.MouseMove, pos, Qt.MouseButton.NoButton, buttons, modifier
    )
    QApplication.sendEvent(widget, event)


def _mouseClick(widget, pos, button, modifier=None):
    _mouseMove(widget, pos)
    _mousePress(widget, pos, button, modifier)
    _mouseRelease(widget, pos, button, modifier)


def _mouseDrag(widget, positions, button, modifier=None):
    _mouseMove(widget, positions[0])
    _mousePress(widget, positions[0], button, modifier)
    # Delay for 10 ms for drag to be recognized
    QTest.qWait(10)
    for pos in positions[1:]:
        _mouseMove(widget, pos, button, modifier)
    _mouseRelease(widget, positions[-1], button, modifier)
