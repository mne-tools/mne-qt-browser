from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication


def _get_n_figs():
    # Wait for a short time to let the Qt loop clean up
    QTest.qWait(100)
    return len(
        [window for window in QApplication.topLevelWindows() if window.isVisible()]
    )


def _close_all():
    if len(QApplication.topLevelWindows()) > 0:
        QApplication.closeAllWindows()


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
