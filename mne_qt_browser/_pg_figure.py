# -*- coding: utf-8 -*-
"""Base classes and functions for 2D browser backends."""

# Authors: Martin Schulz <dev@earthman-music.de>
#
# License: Simplified BSD

import datetime
import gc
import math
import platform
import sys
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from os.path import getsize

import numpy as np
from PyQt5.QtCore import (QEvent, Qt, pyqtSignal, QRunnable,
                          QObject, QThreadPool, QRectF, QLineF, QPoint,
                          QSettings)
from PyQt5.QtGui import (QFont, QIcon, QPixmap, QTransform,
                         QMouseEvent, QImage, QPainter, QPainterPath)
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (QAction, QColorDialog, QComboBox, QDialog,
                             QDockWidget, QDoubleSpinBox, QFormLayout,
                             QGridLayout, QHBoxLayout, QInputDialog,
                             QLabel, QMainWindow, QMessageBox,
                             QPushButton, QScrollBar, QWidget,
                             QStyleOptionSlider, QStyle,
                             QApplication, QGraphicsView, QProgressBar,
                             QVBoxLayout, QLineEdit, QCheckBox, QScrollArea,
                             QGraphicsLineItem, QGraphicsScene, QTextEdit,
                             QSizePolicy, QSpinBox, QDesktopWidget, QSlider)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from pyqtgraph import (AxisItem, GraphicsView, InfLineLabel, InfiniteLine,
                       LinearRegionItem, PlotCurveItem, PlotItem,
                       Point, TextItem, ViewBox, mkBrush,
                       mkPen, setConfigOption, mkQApp, mkColor)
from scipy.stats import zscore

from mne.viz import plot_sensors
from mne.viz.backends._utils import _init_qt_resources
from mne.viz._figure import BrowserBase
from mne.viz.utils import _simplify_float, _merge_annotations
from mne.annotations import _sync_onset
from mne.io.pick import (_DATA_CH_TYPES_ORDER_DEFAULT,
                         channel_indices_by_type, _DATA_CH_TYPES_SPLIT)
from mne.utils import logger, sizeof_fmt, warn

try:
    from pytestqt.exceptions import capture_exceptions
except ImportError:
    logger.debug('If pytest-qt is not installed, the errors from inside '
                 'the Qt-loop will be occluded and it will be harder '
                 'to trace back the cause.')

    @contextmanager
    def capture_exceptions():
        yield []

name = 'pyqtgraph'


def _get_std_icon(icon_name):
    return QApplication.instance().style().standardIcon(
        getattr(QStyle, icon_name))


class RawTraceItem(PlotCurveItem):
    """Graphics-Object for single data trace."""

    def __init__(self, mne, ch_idx, child=False):
        super().__init__(clickable=True)
        self.mne = mne

        # Set default z-value to 1 to be before other items in scene
        self.setZValue(1)

        if self.mne.is_epochs and not child:
            self.bad_trace = RawTraceItem(self.mne, ch_idx, child=True)

        self.set_ch_idx(ch_idx)
        self.update_color()
        self.update_data()

    def update_color(self):
        """Update the color of the trace (depending on ch_type and bad)."""
        if self.isbad and not self.mne.butterfly:
            self.setPen(self.mne.ch_color_bad)
        else:
            self.setPen(self.color)

    def update_range_idx(self):
        """Should be updated when view-range or ch_idx changes."""
        self.range_idx = np.argwhere(self.mne.picks == self.ch_idx)[0][0]

    def update_ypos(self):
        """Should be updated when butterfly is toggled or ch_idx changes."""
        if self.mne.butterfly and self.mne.fig_selection is not None:
            self.ypos = self.mne.selection_ypos_dict[self.ch_idx]
        elif self.mne.fig_selection is not None and \
                self.mne.old_selection == 'Custom':
            self.ypos = self.range_idx + 1
        elif self.mne.butterfly:
            self.ypos = self.mne.butterfly_type_order.index(self.ch_type) + 1
        else:
            self.ypos = self.range_idx + self.mne.ch_start + 1

    def set_ch_idx(self, ch_idx):
        """Sets the channel index and all deriving indices."""
        # The ch_idx is the index of the channel represented by this trace
        # in the channel-order from the unchanged instance (which also picks
        # refer to).
        self.ch_idx = ch_idx
        # The range_idx is the index of the channel represented by this trace
        # in the shown range.
        self.update_range_idx()
        # The order_idx is the index of the channel represented by this trace
        # in the channel-order (defined e.g. by group_by).
        self.order_idx = np.argwhere(self.mne.ch_order == self.ch_idx)[0][0]
        self.ch_name = self.mne.inst.ch_names[ch_idx]
        self.isbad = self.ch_name in self.mne.info['bads']
        self.ch_type = self.mne.ch_types[ch_idx]
        self.color = self.mne.ch_color_assoc[self.ch_name]
        self.update_ypos()

    def update_data(self):
        """Update data (fetch data from self.mne according to self.ch_idx)."""
        if self.mne.is_epochs or (self.mne.clipping is not None and
                                  self.mne.clipping != 'clamp'):
            connect = 'finite'
            skip = False
        else:
            connect = 'all'
            skip = True

        if self.mne.data_precomputed:
            data = self.mne.data[self.order_idx]
        else:
            data = self.mne.data[self.range_idx]

        # Get decim-specific time if enabled
        if self.mne.decim != 1:
            times = self.mne.decim_times[self.mne.decim_data[self.range_idx]]
            data = data[..., ::self.mne.decim_data[self.range_idx]]
        else:
            times = self.mne.times

        self.setData(times, data, connect=connect, skipFiniteCheck=skip,
                     antialias=self.mne.antialiasing)

        self.setPos(0, self.ypos)

    def mouseClickEvent(self, ev):
        """Customize mouse click events."""
        if (not self.clickable or ev.button() != Qt.MouseButton.LeftButton
                or self.mne.annotation_mode):
            ev.ignore()
            return
        if self.mouseShape().contains(ev.pos()):
            ev.accept()
            self.sigClicked.emit(self, ev)

    def get_xdata(self):
        """Get xdata for testing."""
        return self.xData

    def get_ydata(self):
        """Get ydata for testing."""
        return self.yData + self.ypos


class TimeAxis(AxisItem):
    """The X-Axis displaying the time."""

    def __init__(self, mne):
        self.mne = mne
        self._spacing = None
        super().__init__(orientation='bottom')

    def tickValues(self, minVal, maxVal, size):
        """Customize creation of axis values from visible axis range."""
        if self.mne.is_epochs:
            values = self.mne.midpoints[np.argwhere(
                minVal <= self.mne.midpoints <= maxVal)]
            tick_values = [(len(self.mne.inst.times), values)]
            return tick_values
        else:
            # Save _spacing for later use
            self._spacing = self.tickSpacing(minVal, maxVal, size)
            return super().tickValues(minVal, maxVal, size)

    def tickStrings(self, values, scale, spacing):
        """Customize strings of axis values."""
        if self.mne.is_epochs:
            epoch_nums = self.mne.inst.selection
            ts = epoch_nums[np.in1d(self.mne.midpoints, values).nonzero()[0]]
            tick_strings = [str(v) for v in ts]

        elif self.mne.time_format == 'clock':
            meas_date = self.mne.info['meas_date']
            first_time = datetime.timedelta(seconds=self.mne.inst.first_time)

            digits = np.ceil(-np.log10(min(v[0] for v in self._spacing)
                                       ) + 1).astype(int)
            tick_strings = list()
            for val in values:
                val_time = datetime.timedelta(seconds=val) + \
                           first_time + meas_date
                val_str = val_time.strftime('%H:%M:%S')
                if int(val_time.microsecond):
                    val_str += \
                        f'{round(val_time.microsecond * 1e-6, digits)}'[1:]
                tick_strings.append(val_str)
        else:
            tick_strings = super().tickStrings(values, scale, spacing)

        return tick_strings

    def repaint(self):
        """Repaint Time Axis."""
        self.picture = None
        self.update()

    def get_labels(self):
        """Get labels for testing."""
        values = self.tickValues(*self.mne.viewbox.viewRange()[0],
                                 self.mne.xmax)
        labels = list()
        for spacing, vals in values:
            labels += self.tickStrings(vals, 1, spacing)

        return labels


class ChannelAxis(AxisItem):
    """The Y-Axis displaying the channel-names."""

    def __init__(self, main):
        self.main = main
        self.mne = main.mne
        self.ch_texts = dict()
        super().__init__(orientation='left')
        self.style['autoReduceTextSpace'] = False

    def tickValues(self, minVal, maxVal, size):
        """Customize creation of axis values from visible axis range."""
        minVal, maxVal = sorted((minVal, maxVal))
        values = list(range(round(minVal) + 1, round(maxVal)))
        tick_values = [(1, values)]
        return tick_values

    def tickStrings(self, values, scale, spacing):
        """Customize strings of axis values."""
        # Get channel-names
        if self.mne.butterfly and self.mne.fig_selection is not None:
            tick_strings = list(self.main._make_butterfly_selections_dict())
        elif self.mne.butterfly:
            _, ixs, _ = np.intersect1d(_DATA_CH_TYPES_ORDER_DEFAULT,
                                       self.mne.ch_types, return_indices=True)
            ixs.sort()
            tick_strings = np.array(_DATA_CH_TYPES_ORDER_DEFAULT)[ixs]
        else:
            # Get channel-names and by substracting 1 from tick-values
            # since the first channel starts at y=1.
            tick_strings = self.mne.ch_names[
                self.mne.ch_order[[v - 1 for v in values]]]

        return tick_strings

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        """Customize drawing of axis items."""
        super().drawPicture(p, axisSpec, tickSpecs, textSpecs)
        for rect, flags, text in textSpecs:
            if self.mne.butterfly and self.mne.fig_selection is not None:
                p.setPen(mkPen('black'))
            elif self.mne.butterfly:
                p.setPen(mkPen(self.mne.ch_color_dict[text]))
            elif text in self.mne.info['bads']:
                p.setPen(mkPen(self.mne.ch_color_bad))
            else:
                p.setPen(mkPen(self.mne.ch_color_assoc[text]))
            self.ch_texts[text] = ((rect.left(), rect.left() + rect.width()),
                                   (rect.top(), rect.top() + rect.height()))
            p.drawText(rect, int(flags), text)

    def repaint(self):
        """Repaint Channel Axis."""
        self.picture = None
        self.update()

    def mouseClickEvent(self, event):
        """Customize mouse click events."""
        # Clean up channel-texts
        if not self.mne.butterfly:
            self.ch_texts = {k: v for k, v in self.ch_texts.items()
                             if k in [tr.ch_name for tr in self.mne.traces]}
            # Get channel-name from position of channel-description
            ypos = event.scenePos().y()
            ch_name = None
            for ch_name in self.ch_texts:
                ymin, ymax = self.ch_texts[ch_name][1]
                if ymin < ypos < ymax:
                    break

            if ch_name is not None:
                trace = [tr for tr in self.mne.traces
                         if tr.ch_name == ch_name][0]
                if event.button() == Qt.LeftButton:
                    self.main._bad_ch_clicked(trace)
                elif event.button() == Qt.RightButton:
                    self.main._create_ch_context_fig(trace.range_idx)

    def get_labels(self):
        """Get labels for testing."""
        values = self.tickValues(*self.mne.viewbox.viewRange()[1], None)
        labels = self.tickStrings(values[0][1], None, None)

        return labels


class BaseScrollBar(QScrollBar):
    """Base Class for scrolling directly to the clicked position."""

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        """Customize mouse click events.

        Taken from: https://stackoverflow.com/questions/29710327/
        how-to-override-qscrollbar-onclick-default-behaviour
        """
        if event.button() == Qt.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            control = self.style().hitTestComplexControl(
                QStyle.CC_ScrollBar, opt,
                event.pos(), self)
            if (control == QStyle.SC_ScrollBarAddPage or
                    control == QStyle.SC_ScrollBarSubPage):
                # scroll here
                gr = self.style().subControlRect(QStyle.CC_ScrollBar,
                                                 opt,
                                                 QStyle.SC_ScrollBarGroove,
                                                 self)
                sr = self.style().subControlRect(QStyle.CC_ScrollBar,
                                                 opt,
                                                 QStyle.SC_ScrollBarSlider,
                                                 self)
                if self.orientation() == Qt.Horizontal:
                    pos = event.pos().x()
                    sliderLength = sr.width()
                    sliderMin = gr.x()
                    sliderMax = gr.right() - sliderLength + 1
                    if (self.layoutDirection() == Qt.RightToLeft):
                        opt.upsideDown = not opt.upsideDown
                else:
                    pos = event.pos().y()
                    sliderLength = sr.height()
                    sliderMin = gr.y()
                    sliderMax = gr.bottom() - sliderLength + 1
                self.setValue(QStyle.sliderValueFromPosition(
                    self.minimum(), self.maximum(),
                    pos - sliderMin, sliderMax - sliderMin, opt.upsideDown))
                return

        return super().mousePressEvent(event)


class TimeScrollBar(BaseScrollBar):
    """Scrolls through time."""

    def __init__(self, mne):
        super().__init__(Qt.Horizontal)
        self.mne = mne
        self.step_factor = None

        self.setMinimum(0)
        self.setSingleStep(1)
        self.setPageStep(self.mne.scroll_sensitivity)
        self._update_duration()
        self.setFocusPolicy(Qt.WheelFocus)
        # Because valueChanged is needed (captures every input to scrollbar,
        # not just sliderMoved), there has to be made a differentiation
        # between internal and external changes.
        self.external_change = False
        self.valueChanged.connect(self._time_changed)

    def _time_changed(self, value):
        if not self.external_change:
            value /= self.step_factor
            self.mne.plt.setXRange(value, value + self.mne.duration,
                                   padding=0)

    def update_value(self, value):
        """Update value of the ScrollBar."""
        # Mark change as external to avoid setting
        # XRange again in _time_changed.
        self._update_duration()
        self.external_change = True
        self.setValue(int(value * self.step_factor))
        self.external_change = False

    def _update_duration(self):
        new_step_factor = self.mne.scroll_sensitivity / self.mne.duration
        if new_step_factor != self.step_factor:
            self.step_factor = new_step_factor
            new_maximum = int((self.mne.xmax - self.mne.duration)
                              * self.step_factor)
            self.setMaximum(new_maximum)

    def _update_scroll_sensitivity(self):
        self.setPageStep(self.mne.scroll_sensitivity)
        self.update_value(self.value() / self.step_factor)

    def keyPressEvent(self, event):
        """Customize key press events."""
        # Let main handle the keypress
        event.ignore()


class ChannelScrollBar(BaseScrollBar):
    """Scrolls through channels."""

    def __init__(self, mne):
        super().__init__(Qt.Vertical)
        self.mne = mne

        self.setMinimum(0)
        self._update_nchan()
        self.setSingleStep(1)
        self.setFocusPolicy(Qt.WheelFocus)
        # Because valueChanged is needed (captures every input to scrollbar,
        # not just sliderMoved), there has to be made a differentiation
        # between internal and external changes.
        self.external_change = False
        self.valueChanged.connect(self._channel_changed)

    def _channel_changed(self, value):
        if not self.external_change:
            if self.mne.fig_selection:
                label = list(self.mne.ch_selections.keys())[value]
                self.mne.fig_selection._chkbx_changed(label)
            elif not self.mne.butterfly:
                value = min(value, self.mne.ymax - self.mne.n_channels)
                self.mne.plt.setYRange(value, value + self.mne.n_channels + 1,
                                       padding=0)

    def update_value(self, value):
        """Update value of the ScrollBar."""
        # Mark change as external to avoid setting YRange again in
        # _channel_changed.
        self.external_change = True
        self.setValue(value)
        self.external_change = False
        self._update_nchan()

    def _update_nchan(self):
        if self.mne.group_by in ['position', 'selection']:
            self.setPageStep(1)
            self.setMaximum(len(self.mne.ch_selections) - 1)
        else:
            self.setPageStep(self.mne.n_channels)
            self.setMaximum(self.mne.ymax - self.mne.n_channels - 1)

    def keyPressEvent(self, event):
        """Customize key press events."""
        # Let main handle the keypress
        event.ignore()


class OverviewBar(QGraphicsView):
    """
    Provides overview over channels and current visible range.

    Has different modes:
    - channels: Display channel-types
    - zscore: Display channel-wise zscore across time
    """

    def __init__(self, main):
        super().__init__(QGraphicsScene())
        self.main = main
        self.mne = main.mne
        self.bg_img = None
        self.bg_pxmp = None
        self.bg_pxmp_item = None
        # Set minimum Size to 1/10 of display size
        min_h = int(QApplication.desktop().screenGeometry().height() / 10)
        self.setMinimumSize(1, 1)
        self.setFixedHeight(min_h)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.set_background()

        # Initialize Graphics-Items
        # Bad channels
        self.bad_line_dict = dict()
        self.update_bad_channels()

        # Events
        self.event_line_dict = dict()
        self.update_events()

        # Annotations
        self.annotations_rect_dict = dict()
        self.update_annotations()

        # View Range
        self.viewrange_rect = None
        self.update_viewrange()

    def update_bad_channels(self):
        """Update representation of bad channels."""
        bad_set = set(self.mne.info['bads'])
        line_set = set(self.bad_line_dict.keys())

        add_chs = bad_set.difference(line_set)
        rm_chs = line_set.difference(bad_set)

        for line_idx, ch_idx in enumerate(self.mne.ch_order):
            ch_name = self.mne.ch_names[ch_idx]
            if ch_name in add_chs:
                start = self._mapFromData(0, line_idx)
                stop = self._mapFromData(self.mne.inst.times[-1], line_idx)
                pen = mkColor(self.mne.ch_color_bad)
                line = self.scene().addLine(QLineF(start, stop), pen)
                line.setZValue(2)
                self.bad_line_dict[ch_name] = line
            elif ch_name in rm_chs:
                self.scene().removeItem(self.bad_line_dict[ch_name])
                self.bad_line_dict.pop(ch_name)

    def update_events(self):
        if self.mne.event_nums is not None and self.mne.events_visible:
            for ev_t, ev_id in zip(self.mne.event_times, self.mne.event_nums):
                color_name = self.mne.event_color_dict[ev_id]
                color = mkColor(color_name)
                color.setAlpha(100)
                pen = mkPen(color)
                top_left = self._mapFromData(ev_t, 0)
                bottom_right = self._mapFromData(ev_t, len(self.mne.ch_order))
                line = self.scene().addLine(QLineF(top_left, bottom_right),
                                            pen)
                line.setZValue(1)
                self.event_line_dict[ev_t] = line
        else:
            for event_line in self.event_line_dict.values():
                self.scene().removeItem(event_line)
            self.event_line_dict.clear()

    def update_annotations(self):
        """Update representation of annotations."""
        annotations = self.mne.inst.annotations
        # Exclude non-visible annotations
        annot_set = set([annot['onset'] for annot in annotations if
                         self.mne.visible_annotations[annot['description']]])
        rect_set = set(self.annotations_rect_dict.keys())

        add_onsets = annot_set.difference(rect_set)
        rm_onsets = rect_set.difference(annot_set)

        # Add missing onsets
        for add_onset in add_onsets:
            plot_onset = _sync_onset(self.mne.inst, add_onset)
            annot_idx = np.argwhere(self.mne.inst.annotations.onset
                                    == add_onset)[0][0]
            duration = annotations.duration[annot_idx]
            description = annotations.description[annot_idx]
            color_name = self.mne.annotation_segment_colors[description]
            color = mkColor(color_name)
            color.setAlpha(150)
            pen = mkPen(color)
            brush = mkBrush(color)
            top_left = self._mapFromData(plot_onset, 0)
            bottom_right = self._mapFromData(plot_onset + duration,
                                             len(self.mne.ch_order))
            rect = self.scene().addRect(QRectF(top_left, bottom_right),
                                        pen, brush)
            rect.setZValue(3)
            self.annotations_rect_dict[add_onset] = {'rect': rect,
                                                     'plot_onset': plot_onset,
                                                     'duration': duration}

        # Remove onsets
        for rm_onset in rm_onsets:
            self.scene().removeItem(self.annotations_rect_dict[rm_onset]
                                    ['rect'])
            self.annotations_rect_dict.pop(rm_onset)

        # Edit changed duration
        for edit_onset in self.annotations_rect_dict:
            plot_onset = _sync_onset(self.mne.inst, edit_onset)
            annot_idx = np.where(self.mne.inst.annotations.onset
                                 == edit_onset)[0][0]
            duration = annotations.duration[annot_idx]
            rect_duration = self.annotations_rect_dict[edit_onset]['duration']
            if duration != rect_duration:
                self.annotations_rect_dict[edit_onset]['duration'] = duration
                rect = self.annotations_rect_dict[edit_onset]['rect']
                top_left = self._mapFromData(plot_onset, 0)
                bottom_right = self._mapFromData(plot_onset + duration,
                                                 len(self.mne.ch_order))
                rect.setRect(QRectF(top_left, bottom_right))

    def update_viewrange(self):
        if self.mne.butterfly:
            top_left = self._mapFromData(self.mne.t_start, 0)
            bottom_right = self._mapFromData(self.mne.t_start +
                                             self.mne.duration, self.mne.ymax)
        else:
            top_left = self._mapFromData(self.mne.t_start, self.mne.ch_start)
            bottom_right = self._mapFromData(self.mne.t_start
                                             + self.mne.duration,
                                             self.mne.ch_start
                                             + self.mne.n_channels)
        rect = QRectF(top_left, bottom_right)
        if self.viewrange_rect is None:
            pen = mkPen(color='g')
            brush = mkBrush(color=(0, 0, 0, 100))
            self.viewrange_rect = self.scene().addRect(rect, pen, brush)
            self.viewrange_rect.setZValue(4)
        else:
            self.viewrange_rect.setRect(rect)

    def _set_range_from_pos(self, pos):
        x, y = self._mapToData(pos)
        if x == '-offbounds':
            xmin, xmax = (0, self.mne.duration)
        elif x == '+offbounds':
            xmin, xmax = (self.mne.xmax - self.mne.duration, self.mne.xmax)
        else:
            # Move middle of view range to click position
            xmin = x - self.mne.duration / 2
            xmax = xmin + self.mne.duration
        self.mne.plt.setXRange(xmin, xmax, padding=0)

        if y == '-offbounds':
            ymin, ymax = (0, self.mne.n_channels + 1)
        elif y == '+offbounds':
            ymin, ymax = (self.mne.ymax - self.mne.n_channels - 1,
                          self.mne.ymax)
        else:
            ymin = y - self.mne.n_channels / 2
            ymax = ymin + self.mne.n_channels + 1
        if self.mne.fig_selection:
            self.mne.fig_selection._scroll_to_idx(int(ymin))
        else:
            self.mne.plt.setYRange(ymin, ymax, padding=0)

    def mousePressEvent(self, event):
        """Customize mouse press events."""
        self._set_range_from_pos(event.pos())

    def mouseMoveEvent(self, event):
        """Customize mouse move events."""
        self._set_range_from_pos(event.pos())

    def _fit_bg_img(self):
        # Resize Pixmap
        if self.bg_pxmp:
            # Remove previous item from scene
            if (self.bg_pxmp_item is not None and
                    self.bg_pxmp_item in self.scene().items()):
                self.scene().removeItem(self.bg_pxmp_item)

            cnt_rect = self.contentsRect()
            self.bg_pxmp = self.bg_pxmp.scaled(cnt_rect.width(),
                                               cnt_rect.height(),
                                               Qt.IgnoreAspectRatio)
            self.bg_pxmp_item = self.scene().addPixmap(self.bg_pxmp)

    def resizeEvent(self, event):
        """Customize resize event."""
        super().resizeEvent(event)
        cnt_rect = self.contentsRect()
        self.setSceneRect(QRectF(QPoint(0, 0),
                                 QPoint(cnt_rect.width(),
                                        cnt_rect.height())))
        # Resize backgounrd
        self._fit_bg_img()

        # ToDo: This could be improved a lot with view-transforms e.g. with
        #   QGraphicsView.fitInView. The margin-problem could be approached
        #   with https://stackoverflow.com/questions/19640642/
        #   qgraphicsview-fitinview-margins, but came with other problems.
        # Resize Graphics Items (assuming height never changes)
        # Resize bad_channels
        for bad_ch_line in self.bad_line_dict.values():
            current_line = bad_ch_line.line()
            bad_ch_line.setLine(QLineF(current_line.p1(),
                                       Point(cnt_rect.width(),
                                             current_line.y2())))

        # Resize event-lines
        for ev_t, event_line in self.event_line_dict.items():
            top_left = self._mapFromData(ev_t, 0)
            bottom_right = self._mapFromData(ev_t, len(self.mne.ch_order))
            event_line.setLine(QLineF(top_left, bottom_right))

        # Resize annotation-rects
        for annot_dict in self.annotations_rect_dict.values():
            annot_rect = annot_dict['rect']
            plot_onset = annot_dict['plot_onset']
            duration = annot_dict['duration']

            top_left = self._mapFromData(plot_onset, 0)
            bottom_right = self._mapFromData(plot_onset + duration,
                                             len(self.mne.ch_order))
            annot_rect.setRect(QRectF(top_left, bottom_right))

        # Update viewrange-rect
        top_left = self._mapFromData(self.mne.t_start, self.mne.ch_start)
        bottom_right = self._mapFromData(self.mne.t_start
                                         + self.mne.duration,
                                         self.mne.ch_start
                                         + self.mne.n_channels)
        self.viewrange_rect.setRect(QRectF(top_left, bottom_right))

    def set_background(self):
        """Set the background-image for the selected overview-mode."""
        # Add Overview-Pixmap
        if (self.mne.overview_mode == 'channels'
                or not self.mne.enable_precompute):
            channel_rgba = np.empty((len(self.mne.ch_order),
                                     2, 4))
            for line_idx, ch_idx in enumerate(self.mne.ch_order):
                ch_type = self.mne.ch_types[ch_idx]
                color = mkColor(self.mne.ch_color_dict[ch_type])
                channel_rgba[line_idx, :] = color.getRgb()

            channel_rgba = np.require(channel_rgba, np.uint8, 'C')
            self.bg_img = QImage(channel_rgba,
                                 channel_rgba.shape[1],
                                 channel_rgba.shape[0],
                                 QImage.Format_RGBA8888)
            self.bg_pxmp = QPixmap.fromImage(self.bg_img)

        elif self.mne.overview_mode == 'zscore':
            self.bg_img = QImage(self.mne.zscore_rgba,
                                 self.mne.zscore_rgba.shape[1],
                                 self.mne.zscore_rgba.shape[0],
                                 QImage.Format_RGBA8888)
            self.bg_pxmp = QPixmap.fromImage(self.bg_img)

        self._fit_bg_img()

    def _mapFromData(self, x, y):
        # Include padding from black frame
        point_x = self.width() * x / self.mne.xmax
        point_y = self.height() * y / len(self.mne.ch_order)

        return Point(point_x, point_y)

    def _mapToData(self, point):
        # Include padding from black frame
        xnorm = point.x() / self.width()
        if xnorm < 0:
            x = '-offbounds'
        elif xnorm > 1:
            x = '+offbounds'
        else:
            time_idx = int((len(self.mne.inst.times) - 1) * xnorm)
            x = self.mne.inst.times[time_idx]

        ynorm = point.y() / self.height()
        if ynorm < 0:
            y = '-offbounds'
        elif ynorm > 1:
            y = '+offbounds'
        else:
            y = len(self.mne.ch_order) * ynorm

        return x, y

    def keyPressEvent(self, event):
        self.main.keyPressEvent(event)


class RawViewBox(ViewBox):
    """PyQtGraph-Wrapper for interaction with the View."""

    def __init__(self, main):
        super().__init__(invertY=True)
        self.enableAutoRange(enable=False, x=False, y=False)
        self.main = main
        self.mne = main.mne
        self._drag_start = None
        self._drag_region = None

    def mouseDragEvent(self, event, axis=None):
        """Customize mouse drag events."""
        event.accept()

        if event.button() == Qt.LeftButton \
                and self.mne.annotation_mode:
            if self.mne.current_description:
                description = self.mne.current_description
                if event.isStart():
                    self._drag_start = self.mapSceneToView(
                        event.lastScenePos()).x()
                    drag_stop = self.mapSceneToView(event.scenePos()).x()
                    self._drag_region = AnnotRegion(self.mne,
                                                    description=description,
                                                    values=(self._drag_start,
                                                            drag_stop))
                    self.mne.plt.addItem(self._drag_region)
                    self.mne.plt.addItem(self._drag_region.label_item)
                elif event.isFinish():
                    drag_stop = self.mapSceneToView(event.scenePos()).x()
                    self._drag_region.setRegion((self._drag_start, drag_stop))
                    plot_onset = min(self._drag_start, drag_stop)
                    plot_offset = max(self._drag_start, drag_stop)
                    duration = abs(self._drag_start - drag_stop)

                    # Add to annotations
                    onset = _sync_onset(self.mne.inst, plot_onset,
                                        inverse=True)
                    _merge_annotations(onset, onset + duration,
                                       self.mne.current_description,
                                       self.mne.inst.annotations)

                    # Add to regions/merge regions
                    merge_values = [plot_onset, plot_offset]
                    rm_regions = list()
                    for region in [r for r in self.mne.regions
                                   if r.description ==
                                   self.mne.current_description]:
                        values = region.getRegion()
                        if any([plot_onset < val < plot_offset for val in
                                values]):
                            merge_values += values
                            rm_regions.append(region)
                    if len(merge_values) > 2:
                        self._drag_region.setRegion((min(merge_values),
                                                     max(merge_values)))
                    for rm_region in rm_regions:
                        self.main._remove_region(rm_region, from_annot=False)
                    self.main._add_region(plot_onset, duration,
                                          self.mne.current_description,
                                          self._drag_region)
                    self._drag_region.select(True)

                    # Update Overview-Bar
                    self.mne.overview_bar.update_annotations()
                else:
                    self._drag_region.setRegion((self._drag_start,
                                                 self.mapSceneToView(
                                                     event.scenePos()).x()))
            elif event.isFinish():
                QMessageBox.warning(self.main, 'No description!',
                                    'No description is given, add one!')

    def mouseClickEvent(self, event):
        """Customize mouse click events."""
        # If we want the context-menu back, uncomment following line
        # super().mouseClickEvent(event)
        if not self.mne.annotation_mode:
            if event.button() == Qt.LeftButton:
                self.main._add_vline(self.mapSceneToView(
                    event.scenePos()).x())
            elif event.button() == Qt.RightButton:
                self.main._remove_vline()

    def wheelEvent(self, ev, axis=None):
        """Customize mouse wheel/trackpad-scroll events."""
        ev.accept()
        scroll = -1 * ev.delta() / 120
        if ev.orientation() == Qt.Horizontal:
            self.main.hscroll(scroll * 10)
        elif ev.orientation() == Qt.Vertical:
            self.main.vscroll(scroll)

    def keyPressEvent(self, event):
        self.main.keyPressEvent(event)


class VLineLabel(InfLineLabel):
    """Label of the vline displaying the time."""

    def __init__(self, vline):
        super().__init__(vline, text='{value:.3f} s', position=0.98,
                         fill='g', color='b', movable=True)
        self.vline = vline
        self.cursorOffset = None

    def mouseDragEvent(self, ev):
        """Customize mouse drag events."""
        if self.movable and ev.button() == Qt.LeftButton:
            if ev.isStart():
                self.vline.moving = True
                self.cursorOffset = (self.vline.pos() -
                                     self.mapToView(ev.buttonDownPos()))
            ev.accept()

            if not self.vline.moving:
                return

            self.vline.setPos(self.cursorOffset + self.mapToView(ev.pos()))
            self.vline.sigDragged.emit(self)
            if ev.isFinish():
                self.vline.moving = False
                self.vline.sigPositionChangeFinished.emit(self)


class VLine(InfiniteLine):
    """Marker to be placed inside the Trace-Plot."""

    def __init__(self, pos, bounds):
        super().__init__(pos, pen='g', hoverPen='y',
                         movable=True, bounds=bounds)
        self.label = VLineLabel(self)


class EventLine(InfiniteLine):
    """Displays Events inside Trace-Plot"""

    def __init__(self, pos, id, color):
        super().__init__(pos, pen=color, movable=False,
                         label=str(id), labelOpts={'position': 0.98,
                                                   'color': color,
                                                   'anchors': [(0, 0.5),
                                                               (0, 0.5)]})
        self.label.setFont(QFont('AnyStyle', 10, QFont.Bold))
        self.setZValue(0)


class Crosshair(InfiniteLine):
    """Continously updating marker inside the Trace-Plot."""

    def __init__(self):
        super().__init__(angle=90, movable=False, pen='g')
        self.y = 1

    def set_data(self, x, y):
        """Set x and y data for crosshair point."""
        self.setPos(x)
        self.y = y

    def paint(self, p, *args):
        super().paint(p, *args)

        p.setPen(mkPen('r', width=4))
        p.drawPoint(Point(self.y, 0))


class BaseScaleBar:
    def __init__(self, mne, ch_type):
        self.mne = mne
        self.ch_type = ch_type
        self.ypos = None

    def _set_position(self, x, y):
        pass

    def _is_visible(self):
        return self.ch_type in self.mne.ch_types[self.mne.picks]

    def _get_ypos(self):
        if self.mne.butterfly:
            self.ypos = self.mne.butterfly_type_order.index(self.ch_type) + 1
        else:
            ch_type_idxs = np.where(self.mne.ch_types[self.mne.picks]
                                    == self.ch_type)[0]

            for idx in ch_type_idxs:
                ch_name = self.mne.ch_names[self.mne.picks[idx]]
                if ch_name not in self.mne.info['bads'] and \
                        ch_name not in self.mne.whitened_ch_names:
                    self.ypos = self.mne.ch_start + idx + 1
                    break

    def update_x_position(self):
        """Update x-position of Scalebar."""
        if self._is_visible():
            if self.ypos is None:
                self._get_ypos()
            self._set_position(self.mne.t_start, self.ypos)

    def update_y_position(self):
        """Update y-position of Scalebar."""
        if self._is_visible():
            self.setVisible(True)
            self._get_ypos()
            self._set_position(self.mne.t_start, self.ypos)
        else:
            self.setVisible(False)


class ScaleBarText(BaseScaleBar, TextItem):
    def __init__(self, mne, ch_type):
        BaseScaleBar.__init__(self, mne, ch_type)
        TextItem.__init__(self, color='#AA3377')

        self.setFont(QFont('AnyStyle', 10))
        self.setZValue(2)  # To draw over RawTraceItems

        self.update_value()
        self.update_y_position()

    def update_value(self):
        """Update value of ScaleBarText."""
        scaler = 1 if self.mne.butterfly else 2
        inv_norm = (scaler *
                    self.mne.scalings[self.ch_type] *
                    self.mne.unit_scalings[self.ch_type] /
                    self.mne.scale_factor)
        self.setText(f'{_simplify_float(inv_norm)} '
                     f'{self.mne.units[self.ch_type]}')

    def _set_position(self, x, y):
        self.setPos(x, y)


class ScaleBar(BaseScaleBar, QGraphicsLineItem):
    def __init__(self, mne, ch_type):
        BaseScaleBar.__init__(self, mne, ch_type)
        QGraphicsLineItem.__init__(self)

        self.setZValue(1)
        self.setPen(mkPen(color='#AA3377', width=5))
        self.update_y_position()

    def _set_position(self, x, y):
        self.setLine(QLineF(x, y - 0.5, x, y + 0.5))

    def get_ydata(self):
        """Get y-data for tests."""
        line = self.line()
        return line.y1(), line.y2()


class _BaseDialog(QDialog):
    def __init__(self, main, widget=None,
                 modal=False, name=None, title=None):
        super().__init__(main)
        self.main = main
        self.widget = widget
        self.mne = main.mne
        self.name = name
        self.modal = modal

        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self.mne.child_figs.append(self)

        if self.name is not None:
            setattr(self.mne, self.name, self)

        if title is not None:
            self.setWindowTitle(title)

        if self.widget is not None:
            layout = QVBoxLayout()
            layout.addWidget(self.widget)
            self.setLayout(layout)

    def show(self, center=True):
        if self.modal:
            self.open()
        else:
            super().show()

        if center:
            # center dialog
            qr = self.frameGeometry()
            cp = QDesktopWidget().availableGeometry().center()
            qr.moveCenter(cp)
            self.move(qr.topLeft())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            self.parent().keyPressEvent(event)

    def closeEvent(self, event):
        if hasattr(self, 'name') and hasattr(self, 'mne'):
            if self.name is not None and hasattr(self.mne, self.name):
                setattr(self.mne, self.name, None)
            if self in self.mne.child_figs:
                self.mne.child_figs.remove(self)
        event.accept()


class SettingsDialog(_BaseDialog):
    """Shows additional settings."""

    def __init__(self, main, **kwargs):
        super().__init__(main, **kwargs)

        layout = QFormLayout()

        self.downsampling_box = QSpinBox()
        self.downsampling_box.setToolTip('Set an integer as the downsampling'
                                         ' factor or "Auto" to get the factor'
                                         ' from the visible range.\n'
                                         ' Setting the factor 1 means no '
                                         'downsampling.\n'
                                         ' Default is 1.')
        self.downsampling_box.setMinimum(0)
        self.downsampling_box.setSpecialValueText('Auto')
        self.downsampling_box.valueChanged.connect(partial(
            self._value_changed, value_name='downsampling'))
        self.downsampling_box.setValue(0 if self.mne.downsampling == 'auto'
                                       else self.mne.downsampling)
        layout.addRow('downsampling', self.downsampling_box)

        self.ds_method_cmbx = QComboBox()
        self.ds_method_cmbx.setToolTip(
            '<h2>Downsampling Method</h2>'
            '<ul>'
            '<li>subsample:<br>'
            'Only take every n-th sample.</li>'
            '<li>mean:<br>'
            'Take the mean of n samples.</li>'
            '<li>peak:<br>'
            'Draws a saw wave from the minimum to the maximum from a '
            'collection of n samples.</li>'
            '</ul>'
            '<i>(Those methods are adapted from '
            'pyqtgraph)</i><br>'
            'Default is "peak".')
        self.ds_method_cmbx.addItems(['subsample', 'mean', 'peak'])
        self.ds_method_cmbx.currentTextChanged.connect(partial(
            self._value_changed, value_name='ds_method'))
        self.ds_method_cmbx.setCurrentText(
            self.mne.ds_method)
        layout.addRow('ds_method', self.ds_method_cmbx)

        self.scroll_sensitivity_slider = QSlider(Qt.Horizontal)
        self.scroll_sensitivity_slider.setMinimum(10)
        self.scroll_sensitivity_slider.setMaximum(1000)
        self.scroll_sensitivity_slider.setToolTip('Set the sensitivity of '
                                                  'the scrolling in '
                                                  'horizontal direction.')
        self.scroll_sensitivity_slider.valueChanged.connect(partial(
            self._value_changed, value_name='scroll_sensitivity'))
        # Set default
        self.scroll_sensitivity_slider.setValue(self.mne.scroll_sensitivity)
        layout.addRow('horizontal scroll sensitivity',
                      self.scroll_sensitivity_slider)
        self.setLayout(layout)
        self.show()

    def _value_changed(self, new_value, value_name):
        if value_name == 'downsampling' and new_value == 0:
            new_value = 'auto'

        setattr(self.mne, value_name, new_value)
        QSettings().setValue(value_name, new_value)

        if value_name == 'scroll_sensitivity':
            self.mne.ax_hscroll._update_scroll_sensitivity()
        else:
            self.main._redraw()


class HelpDialog(_BaseDialog):
    """Shows all keyboard-shortcuts."""

    def __init__(self, main, **kwargs):
        super().__init__(main, **kwargs)

        # Show all keyboard-shortcuts in a Scroll-Area
        layout = QVBoxLayout()
        scroll_area = QScrollArea()
        scroll_area.setSizePolicy(QSizePolicy.MinimumExpanding,
                                  QSizePolicy.MinimumExpanding)
        scroll_widget = QWidget()
        form_layout = QFormLayout()
        for key in main.mne.keyboard_shortcuts:
            key_dict = main.mne.keyboard_shortcuts[key]
            if 'description' in key_dict:
                if 'alias' in key_dict:
                    key = key_dict['alias']
                for idx, key_des in enumerate(key_dict['description']):
                    key_name = key
                    if 'modifier' in key_dict:
                        mod = key_dict['modifier'][idx]
                        if mod is not None:
                            key_name = mod + ' + ' + key_name
                    form_layout.addRow(key_name, QLabel(key_des))
        scroll_widget.setLayout(form_layout)
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        self.setLayout(layout)
        self.show()


class ProjDialog(_BaseDialog):
    """A dialog to toggle projections."""

    def __init__(self, main, **kwargs):
        self.external_change = True
        # Create projection-layout
        super().__init__(main, **kwargs)

        layout = QVBoxLayout()
        labels = [p['desc'] for p in self.mne.projs]
        for ix, active in enumerate(self.mne.projs_active):
            if active:
                labels[ix] += ' (already applied)'

        # make title
        layout.addWidget(QLabel('Mark projectors applied on the plot.\n'
                                '(Applied projectors are dimmed).'))

        # Add checkboxes
        self.checkboxes = list()
        for idx, label in enumerate(labels):
            chkbx = QCheckBox(label)
            chkbx.setChecked(bool(self.mne.projs_on[idx]))
            chkbx.clicked.connect(partial(self._proj_changed, idx=idx))
            if self.mne.projs_active[idx]:
                chkbx.setEnabled(False)
            self.checkboxes.append(chkbx)
            layout.addWidget(chkbx)

        self.toggle_all_bt = QPushButton('Toggle All')
        self.toggle_all_bt.clicked.connect(self.toggle_all)
        layout.addWidget(self.toggle_all_bt)
        self.setLayout(layout)
        self.show()

    def _proj_changed(self, state, idx):
        # Only change if proj wasn't already applied.
        if not self.mne.projs_active[idx]:
            self.mne.projs_on[idx] = state
            self.main._apply_update_projectors()

    def toggle_all(self):
        """Toggle all projectors."""
        self.main._apply_update_projectors(toggle_all=True)

        # Update all checkboxes
        for idx, chkbx in enumerate(self.checkboxes):
            chkbx.setChecked(bool(self.mne.projs_on[idx]))


class _ChannelFig(FigureCanvasQTAgg):
    def __init__(self, figure):
        self.figure = figure
        super().__init__(figure)
        self.setFocusPolicy(Qt.StrongFocus | Qt.WheelFocus)
        self.setFocus()
        self._lasso_path = None
        # Only update when mouse is pressed
        self.setMouseTracking(False)

    def paintEvent(self, event):
        super().paintEvent(event)
        # Lasso-Drawing doesn't seem to work with mpl, thus it is replicated
        # in Qt.
        if self._lasso_path is not None:
            painter = QPainter(self)
            painter.setPen(mkPen('red', width=2))
            painter.drawPath(self._lasso_path)
            painter.end()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)

        if self._lasso_path is None:
            self._lasso_path = QPainterPath()
            self._lasso_path.moveTo(event.pos())
        else:
            self._lasso_path.lineTo(event.pos())

        self.update()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._lasso_path = None
        self.update()

    def keyPressEvent(self, event):
        event.ignore()


class SelectionDialog(_BaseDialog):
    def __init__(self, main):
        # Create widget
        super().__init__(main, name='fig_selection',
                         title='Channel selection')
        xpos = QApplication.desktop().screenGeometry().width() - 400
        self.setGeometry(xpos, 100, 400, 800)

        layout = QVBoxLayout()

        # Add channel plot
        self.channel_fig = plot_sensors(self.mne.info, kind='select',
                                        ch_type='all', title='',
                                        ch_groups=self.mne.group_by,
                                        show=False)[0]
        self.channel_fig.canvas.mpl_connect('lasso_event',
                                            self._set_custom_selection)
        self.channel_widget = _ChannelFig(self.channel_fig)
        layout.addWidget(self.channel_widget)

        selections_dict = self.mne.ch_selections
        selections_dict.update(Custom=np.array([], dtype=int))  # for lasso

        self.chkbxs = OrderedDict()
        for label in selections_dict:
            chkbx = QCheckBox(label)
            chkbx.clicked.connect(partial(self._chkbx_changed, label))
            self.chkbxs[label] = chkbx
            layout.addWidget(chkbx)

        self.mne.old_selection = list(selections_dict.keys())[0]
        self.chkbxs[self.mne.old_selection].setChecked(True)

        self._update_highlighted_sensors()

        # add instructions at bottom
        instructions = (
            'To use a custom selection, first click-drag on the sensor plot '
            'to "lasso" the sensors you want to select, or hold Ctrl while '
            'clicking individual sensors. Holding Ctrl while click-dragging '
            'allows a lasso selection adding to (rather than replacing) the '
            'existing selection.')
        help_widget = QTextEdit(instructions)
        help_widget.setReadOnly(True)
        layout.addWidget(help_widget)

        self.setLayout(layout)
        self.show(center=False)

    def _chkbx_changed(self, label):
        # Disable butterfly if checkbox is clicked
        if self.mne.butterfly:
            self.main._set_butterfly(False)
        # Disable other checkboxes
        for chkbx in self.chkbxs.values():
            chkbx.setChecked(False)
        if (label == 'Custom' and
                not len(self.mne.ch_selections['Custom'])):
            label = self.mne.old_selection
        # Select the checkbox no matter if clicked on when active or not
        self.chkbxs[label].setChecked(True)
        # Update selections
        self.mne.old_selection = label
        self.mne.picks = np.asarray(self.mne.ch_selections[label])
        self.mne.n_channels = len(self.mne.picks)
        # Update highlighted sensors
        self._update_highlighted_sensors()
        # if "Vertex" is defined, some channels appear twice, so if
        # "Vertex" is selected, ch_start should be the *first* match;
        # otherwise it should be the *last* match (since "Vertex" is
        # always the first selection group, if it exists).
        if label == 'Custom':
            self.mne.ch_start = 0
        else:
            all_values = list()
            for key, chs in self.mne.ch_selections.items():
                if np.array_equal(chs, self.mne.picks):
                    self.mne.ch_start = len(all_values)
                    break
                else:
                    all_values = np.concatenate([all_values, chs])

        # Apply changes on view
        self.mne.plt.setYRange(self.mne.ch_start,
                               self.mne.ch_start + self.mne.n_channels + 1,
                               padding=0)

        # Update scrollbar
        label_idx = list(self.mne.ch_selections.keys()).index(label)
        self.mne.ax_vscroll.update_value(label_idx)

        # Update all y-positions, because channels can appear in multiple
        # selections on different y-positions
        for trace in self.mne.traces:
            trace.update_ypos()
            trace.update_data()

    def _set_custom_selection(self):
        chs = self.channel_fig.lasso.selection
        inds = np.in1d(self.mne.ch_names, chs)
        self.mne.ch_selections['Custom'] = inds.nonzero()[0]
        if any(inds):
            self._chkbx_changed('Custom')

    def _update_highlighted_sensors(self):
        inds = np.in1d(self.mne.fig_selection.channel_fig.lasso.ch_names,
                       self.mne.ch_names[self.mne.picks]).nonzero()[0]
        self.channel_fig.lasso.select_many(inds)
        self.channel_widget.draw()

    def _update_bad_sensors(self, pick, mark_bad):
        sensor_picks = list()
        ch_indices = channel_indices_by_type(self.mne.info)
        for this_type in _DATA_CH_TYPES_SPLIT:
            if this_type in self.mne.ch_types:
                sensor_picks.extend(ch_indices[this_type])
        sensor_idx = np.in1d(sensor_picks, pick).nonzero()[0]
        # change the sensor color
        fig = self.channel_fig
        fig.lasso.ec[sensor_idx, 0] = float(mark_bad)  # change R of RGBA array
        fig.lasso.collection.set_edgecolors(fig.lasso.ec)
        fig.canvas.draw_idle()
        self.channel_widget.draw()

    def _style_butterfly(self):
        for key, chkbx in self.chkbxs.items():
            if self.mne.butterfly:
                chkbx.setChecked(False)
            else:
                if key == self.mne.old_selection:
                    chkbx.setChecked(True)
        self._update_highlighted_sensors()

    def _scroll_selection(self, step):
        name_idx = list(self.mne.ch_selections.keys()).index(
            self.mne.old_selection)
        new_idx = np.clip(name_idx + step,
                          0, len(self.mne.ch_selections) - 1)
        new_label = list(self.mne.ch_selections.keys())[new_idx]
        self._chkbx_changed(new_label)

    def _scroll_to_idx(self, idx):
        all_values = list()
        label = list(self.mne.ch_selections.keys())[0]
        for key, values in self.mne.ch_selections.items():
            all_values = np.concatenate([all_values, values])
            if idx < len(all_values):
                label = key
                break
        self._chkbx_changed(label)

    def closeEvent(self, event):
        super().closeEvent(event)
        if hasattr(self, 'main'):
            self.main.close()


class AnnotRegion(LinearRegionItem):
    """Graphics-Oobject for Annotations."""

    regionChangeFinished = pyqtSignal(object)
    gotSelected = pyqtSignal(object)
    removeRequested = pyqtSignal(object)

    def __init__(self, mne, description, values):
        super().__init__(values=values, orientation='vertical',
                         movable=True, swapMode='sort')
        # Set default z-value to 0 to be behind other items in scene
        self.setZValue(0)

        self.sigRegionChangeFinished.connect(self._region_changed)
        self.mne = mne
        self.description = description
        self.old_onset = values[0]
        self.selected = False

        self.label_item = TextItem(text=description, anchor=(0.5, 0.5))
        self.label_item.setFont(QFont('AnyStyle', 10, QFont.Bold))
        self.sigRegionChanged.connect(self.update_label_pos)

        self.update_color()

    def _region_changed(self):
        self.regionChangeFinished.emit(self)
        self.old_onset = self.getRegion()[0]

    def update_color(self):
        """Update color of annotation-region."""
        color_string = self.mne.annotation_segment_colors[self.description]
        self.base_color = mkColor(color_string)
        self.hover_color = mkColor(color_string)
        self.text_color = mkColor(color_string)
        self.base_color.setAlpha(75)
        self.hover_color.setAlpha(150)
        self.text_color.setAlpha(255)
        self.line_pen = mkPen(color=self.hover_color, width=2)
        self.hover_pen = mkPen(color=self.text_color, width=2)
        self.setBrush(self.base_color)
        self.setHoverBrush(self.hover_color)
        self.label_item.setColor(self.text_color)
        for line in self.lines:
            line.setPen(self.line_pen)
            line.setHoverPen(self.hover_pen)
        self.update()

    def update_description(self, description):
        """Update description of annoation-region."""
        self.description = description
        self.label_item.setText(description)
        self.label_item.update()

    def update_visible(self, visible):
        """Update if annotation-region is visible."""
        self.setVisible(visible)
        self.label_item.setVisible(visible)

    def remove(self):
        """Remove annotation-region."""
        self.removeRequested.emit(self)
        vb = self.mne.viewbox
        if vb and self.label_item in vb.addedItems:
            vb.removeItem(self.label_item)

    def select(self, selected):
        """Update select-state of annotation-region."""
        self.selected = selected
        if selected:
            self.label_item.setColor('w')
            self.label_item.fill = mkBrush(self.hover_color)
            self.gotSelected.emit(self)
        else:
            self.label_item.setColor(self.text_color)
            self.label_item.fill = mkBrush(None)
        self.label_item.update()

    def mouseClickEvent(self, event):
        """Customize mouse click events."""
        if self.mne.annotation_mode:
            if event.button() == Qt.LeftButton and self.movable:
                self.select(True)
                event.accept()
            elif event.button() == Qt.RightButton and self.movable:
                self.remove()
                # Propagate remove request to lower annotations if overlapping
                event.ignore()
        else:
            event.ignore()

    def update_label_pos(self):
        """Update position of description-label from annotation-region."""
        rgn = self.getRegion()
        vb = self.mne.viewbox
        if vb:
            ymax = vb.viewRange()[1][1]
            self.label_item.setPos(sum(rgn) / 2, ymax - 0.3)


class _AnnotEditDialog(_BaseDialog):
    def __init__(self, annot_dock):
        super().__init__(annot_dock.main, title='Edit Annotations')
        self.ad = annot_dock

        self.current_mode = None
        self.curr_des = None

        layout = QVBoxLayout()
        self.descr_label = QLabel()
        if self.mne.selected_region:
            self.mode_cmbx = QComboBox()
            self.mode_cmbx.addItems(['group', 'current'])
            self.mode_cmbx.currentTextChanged.connect(self._mode_changed)
            layout.addWidget(QLabel('Edit Scope:'))
            layout.addWidget(self.mode_cmbx)
        # Set group as default
        self._mode_changed('group')

        layout.addWidget(self.descr_label)
        self.input_w = QLineEdit()
        layout.addWidget(self.input_w)
        bt_layout = QHBoxLayout()
        ok_bt = QPushButton('Ok')
        ok_bt.clicked.connect(self._edit)
        bt_layout.addWidget(ok_bt)
        cancel_bt = QPushButton('Cancel')
        cancel_bt.clicked.connect(self.close)
        bt_layout.addWidget(cancel_bt)
        layout.addLayout(bt_layout)
        self.setLayout(layout)
        self.show()

    def _mode_changed(self, mode):
        self.current_mode = mode
        if mode == 'group':
            curr_des = self.ad.description_cmbx.currentText()
        else:
            curr_des = self.mne.selected_region.description
        self.descr_label.setText(f'Change "{curr_des}" to:')
        self.curr_des = curr_des

    def _edit(self):
        new_des = self.input_w.text()
        if new_des:
            if self.current_mode == 'group' or self.mne.selected_region is \
                    None:
                edit_regions = [r for r in self.mne.regions
                                if r.description == self.curr_des]
                for ed_region in edit_regions:
                    idx = self.main._get_onset_idx(
                        ed_region.getRegion()[0])
                    self.mne.inst.annotations.description[idx] = new_des
                    ed_region.update_description(new_des)
                self.mne.new_annotation_labels.remove(self.curr_des)
                self.mne.new_annotation_labels = \
                    self.main._get_annotation_labels()
                self.mne.visible_annotations[new_des] = \
                    self.mne.visible_annotations.pop(self.curr_des)
                self.mne.annotation_segment_colors[new_des] = \
                    self.mne.annotation_segment_colors.pop(
                        self.curr_des)
            else:
                idx = self.main._get_onset_idx(
                    self.mne.selected_region.getRegion()[0])
                self.mne.inst.annotations.description[idx] = new_des
                self.mne.selected_region.update_description(new_des)
                if new_des not in self.mne.new_annotation_labels:
                    self.mne.new_annotation_labels.append(new_des)
                self.mne.visible_annotations[new_des] = \
                    self.mne.visible_annotations[self.curr_des]
                self.mne.annotation_segment_colors[new_des] = \
                    self.mne.annotation_segment_colors[self.curr_des]
                if self.curr_des not in \
                        self.mne.inst.annotations.description:
                    self.mne.new_annotation_labels.remove(
                        self.curr_des)
                    self.mne.visible_annotations.pop(self.curr_des)
                    self.mne.annotation_segment_colors.pop(
                        self.curr_des)
            self.mne.current_description = new_des
            self.main._setup_annotation_colors()
            self.ad._update_description_cmbx()
            self.ad._update_regions_colors()
            self.close()


class AnnotationDock(QDockWidget):
    """Dock-Window for Management of annotations."""

    def __init__(self, main):
        super().__init__('Annotations')
        self.main = main
        self.mne = main.mne
        self._init_ui()

        self.setFeatures(QDockWidget.DockWidgetMovable |
                         QDockWidget.DockWidgetFloatable)

    def _init_ui(self):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignLeft)

        self.description_cmbx = QComboBox()
        self.description_cmbx.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.description_cmbx.activated.connect(self._description_changed)
        self._update_description_cmbx()
        layout.addWidget(self.description_cmbx)

        add_bt = QPushButton('Add Description')
        add_bt.clicked.connect(self._add_description_dlg)
        layout.addWidget(add_bt)

        rm_bt = QPushButton('Remove Description')
        rm_bt.clicked.connect(self._remove_description)
        layout.addWidget(rm_bt)

        edit_bt = QPushButton('Edit Description')
        edit_bt.clicked.connect(self._edit_description)
        layout.addWidget(edit_bt)

        # Uncomment when custom colors for annotations are implemented in
        # MNE-Python.
        # color_bt = QPushButton('Edit Color')
        # color_bt.clicked.connect(self._set_color)
        # layout.addWidget(color_bt)

        select_bt = QPushButton('Select Visible')
        select_bt.clicked.connect(self._select_annotations)
        layout.addWidget(select_bt)

        # Determine reasonable time decimals from sampling frequency.
        time_decimals = int(np.ceil(np.log10(self.mne.info['sfreq'])))

        layout.addWidget(QLabel('Start:'))
        self.start_bx = QDoubleSpinBox()
        self.start_bx.setDecimals(time_decimals)
        self.start_bx.editingFinished.connect(self._start_changed)
        layout.addWidget(self.start_bx)

        layout.addWidget(QLabel('Stop:'))
        self.stop_bx = QDoubleSpinBox()
        self.stop_bx.setDecimals(time_decimals)
        self.stop_bx.editingFinished.connect(self._stop_changed)
        layout.addWidget(self.stop_bx)

        help_bt = QPushButton(_get_std_icon('SP_DialogHelpButton'), 'Help')
        help_bt.clicked.connect(self._show_help)
        layout.addWidget(help_bt)

        widget.setLayout(layout)
        self.setWidget(widget)

    def _add_description_to_cmbx(self, description):
        color_pixmap = QPixmap(25, 25)
        color = mkColor(self.mne.annotation_segment_colors[description])
        color.setAlpha(75)
        color_pixmap.fill(color)
        color_icon = QIcon(color_pixmap)
        self.description_cmbx.addItem(color_icon, description)

    def _add_description(self, new_description):
        self.mne.new_annotation_labels.append(new_description)
        self.mne.visible_annotations[new_description] = True
        self.main._setup_annotation_colors()
        self._add_description_to_cmbx(new_description)
        self.mne.current_description = new_description
        self.description_cmbx.setCurrentText(new_description)

    def _add_description_dlg(self):
        new_description, ok = QInputDialog.getText(self,
                                                   'Set new description!',
                                                   'New description: ')
        if ok and new_description \
                and new_description not in self.mne.new_annotation_labels:
            self._add_description(new_description)

    def _edit_description(self):
        if len(self.mne.inst.annotations.description) > 0:
            _AnnotEditDialog(self)
        else:
            QMessageBox.information(self, 'No Annotations!',
                                    'There are no annotations yet to edit!')

    def _remove_description(self):
        rm_description = self.description_cmbx.currentText()
        existing_annot = list(self.mne.inst.annotations.description).count(
            rm_description)
        if existing_annot > 0:
            ans = QMessageBox.question(self,
                                       f'Remove annotations '
                                       f'with {rm_description}?',
                                       f'There exist {existing_annot} '
                                       f'annotations with '
                                       f'"{rm_description}".\n'
                                       f'Do you really want to remove them?')
            if ans == QMessageBox.Yes:
                rm_idxs = np.where(
                    self.mne.inst.annotations.description == rm_description)
                for idx in rm_idxs:
                    self.mne.inst.annotations.delete(idx)
                for rm_region in [r for r in self.mne.regions
                                  if r.description == rm_description]:
                    rm_region.remove()
            else:
                return

        # Remove from descriptions
        self.mne.new_annotation_labels.remove(rm_description)
        self._update_description_cmbx()

        # Remove from visible annotations
        self.mne.visible_annotations.pop(rm_description)

        # Remove from color-mapping
        if rm_description in self.mne.annotation_segment_colors:
            self.mne.annotation_segment_colors.pop(rm_description)

        # Set first description in Combo-Box to current description
        if self.description_cmbx.count() > 0:
            self.description_cmbx.setCurrentIndex(0)
            self.mne.current_description = \
                self.description_cmbx.currentText()

    def _select_annotations(self):
        def _set_visible_region(state, description):
            self.mne.visible_annotations[description] = bool(state)

        def _select_all():
            for chkbx in chkbxs:
                chkbx.setChecked(True)

        def _clear_all():
            for chkbx in chkbxs:
                chkbx.setChecked(False)

        select_dlg = QDialog(self)
        chkbxs = list()
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Select visible labels:'))

        # Add descriptions to scroll-area to be scalable.
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        for des in self.mne.visible_annotations:
            chkbx = QCheckBox(des)
            chkbx.setChecked(self.mne.visible_annotations[des])
            chkbx.stateChanged.connect(partial(_set_visible_region,
                                               description=des))
            chkbxs.append(chkbx)
            scroll_layout.addWidget(chkbx)

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        bt_layout = QGridLayout()

        all_bt = QPushButton('All')
        all_bt.clicked.connect(_select_all)
        bt_layout.addWidget(all_bt, 0, 0)

        clear_bt = QPushButton('Clear')
        clear_bt.clicked.connect(_clear_all)
        bt_layout.addWidget(clear_bt, 0, 1)

        ok_bt = QPushButton('Ok')
        ok_bt.clicked.connect(select_dlg.close)
        bt_layout.addWidget(ok_bt, 1, 0, 1, 2)

        layout.addLayout(bt_layout)

        select_dlg.setLayout(layout)
        select_dlg.exec()

        self.main._update_regions_visible()

    def _description_changed(self, descr_idx):
        new_descr = self.description_cmbx.itemText(descr_idx)
        self.mne.current_description = new_descr

    def _start_changed(self):
        start = self.start_bx.value()
        sel_region = self.mne.selected_region
        if sel_region:
            stop = sel_region.getRegion()[1]
            if start < stop:
                sel_region.setRegion((start, stop))
            else:
                QMessageBox.warning(self, 'Invalid value!',
                                    'Start can\'t be bigger or equal to Stop!')
                self.start_bx.setValue(sel_region.getRegion()[0])

    def _stop_changed(self):
        stop = self.stop_bx.value()
        sel_region = self.mne.selected_region
        if sel_region:
            start = sel_region.getRegion()[0]
            if start < stop:
                sel_region.setRegion((start, stop))
            else:
                QMessageBox.warning(self, 'Invalid value!',
                                    'Stop can\'t be smaller '
                                    'or equal to Start!')
                self.stop_bx.setValue(sel_region.getRegion()[1])

    def _set_color(self):
        curr_descr = self.description_cmbx.currentText()
        if curr_descr in self.mne.annotation_segment_colors:
            curr_col = self.mne.annotation_segment_colors[curr_descr]
        else:
            curr_col = None
        color = QColorDialog.getColor(mkColor(curr_col), self,
                                      f'Choose color for {curr_descr}!')
        if color.isValid():
            self.mne.annotation_segment_colors[curr_descr] = color
            self._update_description_cmbx()
            self._update_regions_colors()
            self.mne.overview_bar.update_annotations()

    def update_values(self, region):
        """Update spinbox-values from region."""
        rgn = region.getRegion()
        self.start_bx.setValue(rgn[0])
        self.stop_bx.setValue(rgn[1])

    def _update_description_cmbx(self):
        self.description_cmbx.clear()
        descriptions = self.main._get_annotation_labels()
        for description in descriptions:
            self._add_description_to_cmbx(description)
        self.description_cmbx.setCurrentText(self.mne.current_description)

    def _update_regions_colors(self):
        for region in self.mne.regions:
            region.update_color()

    def reset(self):
        """Reset to default state."""
        if self.description_cmbx.count() > 0:
            self.description_cmbx.setCurrentIndex(0)
            self.mne.current_description = self.description_cmbx.currentText()
        self.start_bx.setValue(0)
        self.stop_bx.setValue(0)

    def _show_help(self):
        QMessageBox.information(self, 'Annotations-Help',
                                '<h1>Help</h1>'
                                '<h2>Annotations</h2>'
                                '<h3>Add Annotations</h3>'
                                'Drag inside the data-view to create '
                                'annotations with the description currently '
                                'selected (leftmost item of the toolbar).'
                                'If there is no description yet, add one '
                                'with the button "Add description".'
                                '<h3>Remove Annotations</h3>'
                                'You can remove single annotations by '
                                'right-clicking on them.'
                                '<h3>Edit Annotations</h3>'
                                'You can edit annotations by dragging them or '
                                'their boundaries. Or you can use the dials '
                                'in the toolbar to adjust the boundaries for '
                                'the current selected annotation.'
                                '<h2>Descriptions</h2>'
                                '<h3>Add Description</h3>'
                                'Add a new description with the button'
                                '"Add description".'
                                '<h3>Edit Description</h3>'
                                'You can edit the description of one single '
                                'annotation or all annotations of the '
                                'currently selected kind with the button '
                                '"Edit description".'
                                '<h3>Remove Description</h3>'
                                'You can remove all annotations of the '
                                'currently selected kind with the button '
                                '"Remove description".')


class BrowserView(GraphicsView):
    """Customized View as part of GraphicsView-Framework."""

    def __init__(self, plot, **kwargs):
        super().__init__(**kwargs)
        self.setCentralItem(plot)
        self.viewport().setAttribute(Qt.WA_AcceptTouchEvents, True)

        self.viewport().grabGesture(Qt.PinchGesture)
        self.viewport().grabGesture(Qt.SwipeGesture)

    # def viewportEvent(self, event):
    #     """Customize viewportEvent for touch-gestures (WIP)."""
    #     if event.type() in [QEvent.TouchBegin, QEvent.TouchUpdate,
    #                         QEvent.TouchEnd]:
    #         if event.touchPoints() == 2:
    #             pass
    #     elif event.type() == QEvent.Gesture:
    #         print('Gesture')
    #     return super().viewportEvent(event)

    def mouseMoveEvent(self, ev):
        """Customize MouseMoveEvent."""
        # Don't set GraphicsView.mouseEnabled to True,
        # we only want part of the functionality pyqtgraph offers here.
        super().mouseMoveEvent(ev)
        self.sigSceneMouseMoved.emit(ev.pos())


class LoadRunnerSignals(QObject):
    """Signals for the LoadRunner (QRunnables aren't QObjects)."""

    loadProgress = pyqtSignal(int)
    processText = pyqtSignal(str)
    loadingFinished = pyqtSignal()


class LoadRunner(QRunnable):
    """A QRunnable for precomputing in a separate QThread."""

    def __init__(self, browser):
        super().__init__()
        self.browser = browser
        self.mne = browser.mne
        self.sigs = LoadRunnerSignals()

    def run(self):
        """Load and process data in a separate QThread."""
        # Split data loading into 10 chunks to show user progress.
        # Testing showed that e.g. n_chunks=100 extends loading time
        # (at least for the sample dataset)
        # because of the frequent gui-update-calls.
        # Thus n_chunks = 10 should suffice.
        data = None
        times = None
        n_chunks = 10
        if not self.mne.is_epochs:
            chunk_size = len(self.browser.mne.inst) // n_chunks
            for n in range(n_chunks):
                start = n * chunk_size
                if n == n_chunks - 1:
                    # Get last chunk which may be larger due to rounding above
                    stop = None
                else:
                    stop = start + chunk_size
                # Load data
                data_chunk, times_chunk = self.browser._load_data(start, stop)
                if data is None:
                    data = data_chunk
                    times = times_chunk
                else:
                    data = np.concatenate((data, data_chunk), axis=1)
                    times = np.concatenate((times, times_chunk), axis=0)
                self.sigs.loadProgress.emit(n + 1)
        else:
            self.browser._load_data()
            self.sigs.loadProgress.emit(n_chunks)

        picks = self.browser.mne.ch_order
        # Deactive remove dc because it will be removed for visible range
        stashed_remove_dc = self.mne.remove_dc
        self.mne.remove_dc = False
        data = self.browser._process_data(data, 0, len(data), picks,
                                          self.sigs)
        self.mne.remove_dc = stashed_remove_dc

        self.browser.mne.global_data = data
        self.browser.mne.global_times = times

        # Calculate Z-Scores
        self.sigs.processText.emit('Calculating Z-Scores...')
        self.browser._get_zscore(data)

        self.sigs.loadingFinished.emit()


class _PGMetaClass(type(BrowserBase), type(QMainWindow)):
    """Class is necessary to prevent a metaclass conflict.

    The conflict arises due to the different types of QMainWindow and
    BrowserBase.
    """

    pass


class PyQtGraphBrowser(BrowserBase, QMainWindow, metaclass=_PGMetaClass):
    """A PyQtGraph-backend for 2D data browsing."""
    gotClosed = pyqtSignal()

    def __init__(self, **kwargs):
        self.backend_name = 'pyqtgraph'

        BrowserBase.__init__(self, **kwargs)
        QMainWindow.__init__(self)

        if self.mne.window_title is not None:
            self.setWindowTitle(self.mne.window_title)

        # Initialize attributes which are only used by pyqtgraph, not by
        # matplotlib and add them to MNEBrowseParams.
        self.mne.fig_settings = None
        self.mne.decim_data = None
        self.mne.decim_times = None
        self.mne.selection_ypos_dict = dict()
        self.mne.enable_precompute = False
        self.mne.data_precomputed = False
        self.mne.show_overview_bar = True
        self.mne.overview_mode = 'channels'
        self.mne.zscore_rgba = None
        self.mne.antialiasing = False
        self.mne.scroll_sensitivity = 100  # Steps per view (relative to time)
        self.mne.downsampling = 1
        self.mne.ds_method = 'peak'

        # Load from QSettings if available
        qsettings_params = ['antialiasing', 'scroll_sensitivity',
                            'downsampling', 'ds_method']
        for qparam in qsettings_params:
            qvalue = QSettings().value(qparam, defaultValue=None)
            if qvalue in ['true', 'false']:
                qvalue = bool(qvalue)
            if qvalue is not None:
                setattr(self.mne, qvalue)

        # Initialize channel-colors for faster indexing later
        self.mne.ch_color_assoc = dict()
        for idx, ch_name in enumerate(self.mne.ch_names):
            ch_type = self.mne.ch_types[idx]
            self.mne.ch_color_assoc[ch_name] = self.mne.ch_color_dict[ch_type]

        # Add Load-Progressbar for loading in a thread
        self.mne.load_prog_label = QLabel('Loading...')
        self.statusBar().addWidget(self.mne.load_prog_label)
        self.mne.load_prog_label.hide()
        self.mne.load_progressbar = QProgressBar()
        # Set to n_chunks of LoadRunner
        self.mne.load_progressbar.setMaximum(10)
        self.statusBar().addWidget(self.mne.load_progressbar, stretch=1)
        self.mne.load_progressbar.hide()

        self.mne.traces = list()
        self.mne.scale_factor = 1
        self.mne.butterfly_type_order = [tp for tp in
                                         _DATA_CH_TYPES_ORDER_DEFAULT
                                         if tp in self.mne.ch_types]

        # Initialize annotations (ToDo: Adjust to MPL)
        self.mne.annotation_mode = False
        self.mne.annotations_visible = True
        self.mne.new_annotation_labels = self._get_annotation_labels()
        if len(self.mne.new_annotation_labels) > 0:
            self.mne.current_description = self.mne.new_annotation_labels[0]
        else:
            self.mne.current_description = None
        self._setup_annotation_colors()
        self.mne.regions = list()
        self.mne.selected_region = None

        # Create centralWidget and layout
        widget = QWidget()
        layout = QGridLayout()

        # Initialize Axis-Items
        time_axis = TimeAxis(self.mne)
        time_axis.setLabel(text='Time', units='s')
        channel_axis = ChannelAxis(self)
        viewbox = RawViewBox(self)
        vars(self.mne).update(time_axis=time_axis, channel_axis=channel_axis,
                              viewbox=viewbox)

        # Start precomputing if enabled
        self._init_precompute()

        # Initialize data (needed in RawTraceItem.update_data).
        self._update_data()

        # Initialize Trace-Plot
        plt = PlotItem(viewBox=viewbox,
                       axisItems={'bottom': time_axis, 'left': channel_axis})
        # Hide AutoRange-Button
        plt.hideButtons()
        # Configure XY-Range
        self.mne.xmax = self.mne.inst.times[-1]
        plt.setXRange(0, self.mne.duration, padding=0)
        # Add one empty line as padding at top (y=0).
        # Negative Y-Axis to display channels from top.
        self.mne.ymax = len(self.mne.ch_order) + 1
        plt.setYRange(0, self.mne.n_channels + 1, padding=0)
        plt.setLimits(xMin=0, xMax=self.mne.xmax,
                      yMin=0, yMax=self.mne.ymax)
        # Connect Signals from PlotItem
        plt.sigXRangeChanged.connect(self._xrange_changed)
        plt.sigYRangeChanged.connect(self._yrange_changed)
        vars(self.mne).update(plt=plt)

        # Add traces
        for ch_idx in self.mne.picks:
            self._add_trace(ch_idx)

        # Add events (add all once, since their representation is simple
        # they shouldn't have a big impact on performance when showing them
        # is handled by QGraphicsView).
        if self.mne.event_nums is not None:
            self.mne.events_visible = True
            for ev_time, ev_id in zip(self.mne.event_times,
                                      self.mne.event_nums):
                color = self.mne.event_color_dict[ev_id]
                event_line = EventLine(ev_time, ev_id, color)
                self.mne.event_lines.append(event_line)

                if 0 < ev_time < self.mne.duration:
                    self.mne.plt.addItem(event_line)
        else:
            self.mne.events_visible = False

        # Add Scale-Bars
        self._add_scalebars()

        # Check for OpenGL
        if self.mne.use_opengl:
            try:
                import OpenGL
            except (ModuleNotFoundError, ImportError):
                warn('PyOpenGL was not found and OpenGL can\'t be used!\n'
                     'Consider installing pyopengl with "pip install pyopengl"'
                     '.')
                self.mne.use_opengl = False
            else:
                logger.info(
                    f'Using pyopengl with version {OpenGL.__version__}')
        # Initialize BrowserView (inherits QGraphicsView)
        view = BrowserView(plt, background='w',
                           useOpenGL=self.mne.use_opengl)
        layout.addWidget(view, 0, 0)

        # Initialize Scroll-Bars
        ax_hscroll = TimeScrollBar(self.mne)
        layout.addWidget(ax_hscroll, 1, 0, 1, 2)

        ax_vscroll = ChannelScrollBar(self.mne)
        layout.addWidget(ax_vscroll, 0, 1)

        # OverviewBar
        overview_bar = OverviewBar(self)
        layout.addWidget(overview_bar, 2, 0, 1, 2)

        # Add Combobox to select Overview-Mode
        self.overview_mode_chkbx = QComboBox()
        self.overview_mode_chkbx.addItems(['channels'])
        self.overview_mode_chkbx.setToolTip('<h2>Overview-Modes</h2>'
                                            '<ul>'
                                            '<li>channels:<br> '
                                            'Display each channel with its '
                                            'channel-type color.</li>'
                                            '<li>zscore:<br>'
                                            ' Display the zscore for the '
                                            'data from each channel across '
                                            'time. '
                                            'Red indicates high z - scores,'
                                            'Blue indicates low z - scores '
                                            ' whilethe boundaries of the '
                                            ' color gradientare defined by '
                                            ' the minimum/maximum z-score. '
                                            'This only works if precompute '
                                            'is set to "True" or it is '
                                            'enabled with "auto" and enough '
                                            'free RAM.</li>'
                                            '</ul>')
        if self.mne.enable_precompute:
            self.overview_mode_chkbx.addItems(['zscore'])
        self.overview_mode_chkbx.currentTextChanged.connect(
            self._overview_mode_changed)
        self.overview_mode_chkbx.setCurrentIndex(0)
        # Avoid taking keyboard-focus
        self.overview_mode_chkbx.setFocusPolicy(Qt.NoFocus)
        overview_mode_layout = QHBoxLayout()
        overview_mode_layout.addWidget(QLabel('Overview-Mode:'))
        overview_mode_layout.addWidget(self.overview_mode_chkbx)
        overview_mode_widget = QWidget()
        overview_mode_widget.setLayout(overview_mode_layout)
        self.statusBar().addPermanentWidget(overview_mode_widget)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Initialize Annotation-Dock
        fig_annotation = AnnotationDock(self)
        self.addDockWidget(Qt.TopDockWidgetArea, fig_annotation)
        fig_annotation.setVisible(False)
        vars(self.mne).update(fig_annotation=fig_annotation)

        # Add annotations as regions
        for annot in self.mne.inst.annotations:
            plot_onset = _sync_onset(self.mne.inst, annot['onset'])
            duration = annot['duration']
            description = annot['description']
            self._add_region(plot_onset, duration, description)

        # Initialize annotations
        self._change_annot_mode()

        # Initialize VLine
        self.mne.vline = None
        self.mne.vline_visible = False

        # Initialize crosshair (as in pyqtgraph example)
        self.mne.crosshair_enabled = False
        self.mne.crosshair_h = None
        self.mne.crosshair = None
        view.sigSceneMouseMoved.connect(self._mouse_moved)

        # Initialize Toolbar
        toolbar = self.addToolBar('Tools')
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        adecr_time = QAction(_get_std_icon('SP_ArrowDown'),
                             'Time', parent=self)
        adecr_time.triggered.connect(partial(self.change_duration, -0.2))
        toolbar.addAction(adecr_time)

        aincr_time = QAction(_get_std_icon('SP_ArrowUp'),
                             'Time', parent=self)
        aincr_time.triggered.connect(partial(self.change_duration, 0.25))
        toolbar.addAction(aincr_time)

        adecr_nchan = QAction(_get_std_icon('SP_ArrowDown'),
                              'Channels', parent=self)
        adecr_nchan.triggered.connect(partial(self.change_nchan, -10))
        toolbar.addAction(adecr_nchan)

        aincr_nchan = QAction(_get_std_icon('SP_ArrowUp'),
                              'Channels', parent=self)
        aincr_nchan.triggered.connect(partial(self.change_nchan, 10))
        toolbar.addAction(aincr_nchan)

        adecr_nchan = QAction(_get_std_icon('SP_ArrowDown'),
                              'Zoom', parent=self)
        adecr_nchan.triggered.connect(partial(self.scale_all, 4 / 5))
        toolbar.addAction(adecr_nchan)

        aincr_nchan = QAction(_get_std_icon('SP_ArrowUp'),
                              'Zoom', parent=self)
        aincr_nchan.triggered.connect(partial(self.scale_all, 5 / 4))
        toolbar.addAction(aincr_nchan)

        atoggle_annot = QAction(_get_std_icon('SP_DialogResetButton'),
                                'Annotations', parent=self)
        atoggle_annot.triggered.connect(self._toggle_annotation_fig)
        toolbar.addAction(atoggle_annot)

        atoggle_proj = QAction(_get_std_icon('SP_DialogOkButton'),
                               'SSP', parent=self)
        atoggle_proj.triggered.connect(self._toggle_proj_fig)
        toolbar.addAction(atoggle_proj)

        atoggle_fullscreen = QAction(_get_std_icon('SP_TitleBarMaxButton'),
                                     'Full-Screen', parent=self)
        atoggle_fullscreen.triggered.connect(self._toggle_fullscreen)
        toolbar.addAction(atoggle_fullscreen)

        asettings = QAction(_get_std_icon('SP_FileDialogDetailedView'),
                            'Settings', parent=self)
        asettings.triggered.connect(self._toggle_settings_fig)
        toolbar.addAction(asettings)

        ahelp = QAction(_get_std_icon('SP_DialogHelpButton'),
                        'Shortcuts', parent=self)
        ahelp.triggered.connect(self._toggle_help_fig)
        toolbar.addAction(ahelp)

        # Add GUI-Elements to MNEBrowserParams-Instance
        vars(self.mne).update(
            plt=plt, view=view, ax_hscroll=ax_hscroll, ax_vscroll=ax_vscroll,
            overview_bar=overview_bar, fig_annotation=fig_annotation,
            toolbar=toolbar
        )

        # Set Size
        width = int(self.mne.figsize[0] * self.logicalDpiX())
        height = int(self.mne.figsize[1] * self.logicalDpiY())
        self.resize(width, height)

        # Initialize Keyboard-Shortcuts
        is_mac = platform.system() == 'Darwin'
        dur_keys = ('fn + ←', 'fn + →') if is_mac else ('Home', 'End')
        ch_keys = ('fn + ↑', 'fn + ↓') if is_mac else ('Page up', 'Page down')
        self.mne.keyboard_shortcuts = {
            'left': {
                'alias': '←',
                'qt_key': Qt.Key_Left,
                'modifier': [None, 'Shift'],
                'slot': [self.hscroll],
                'parameter': [-40, '-full'],
                'description': ['Move left',
                                'Move left (tiny step)']
            },
            'right': {
                'alias': '→',
                'qt_key': Qt.Key_Right,
                'modifier': [None, 'Shift'],
                'slot': [self.hscroll],
                'parameter': [40, '+full'],
                'description': ['Move right',
                                'Move right (tiny step)']
            },
            'up': {
                'alias': '↑',
                'qt_key': Qt.Key_Up,
                'slot': [self.vscroll],
                'parameter': ['-full'],
                'description': ['Move up (page)']
            },
            'down': {
                'alias': '↓',
                'qt_key': Qt.Key_Down,
                'slot': [self.vscroll],
                'parameter': ['+full'],
                'description': ['Move down (page)']
            },
            'home': {
                'alias': dur_keys[0],
                'qt_key': Qt.Key_Home,
                'slot': [self.change_duration],
                'parameter': [-0.2],
                'description': ['Decrease duration']
            },
            'end': {
                'alias': dur_keys[1],
                'qt_key': Qt.Key_End,
                'slot': [self.change_duration],
                'parameter': [0.25],
                'description': ['Increase duration']
            },
            'pagedown': {
                'alias': ch_keys[0],
                'qt_key': Qt.Key_PageDown,
                'modifier': [None, 'Shift'],
                'slot': [self.change_nchan],
                'parameter': [-1, -10],
                'description': ['Decrease shown channels',
                                'Decrease shown channels (tiny step)']
            },
            'pageup': {
                'alias': ch_keys[1],
                'qt_key': Qt.Key_PageUp,
                'modifier': [None, 'Shift'],
                'slot': [self.change_nchan],
                'parameter': [1, 10],
                'description': ['Increase shown channels',
                                'Increase shown channels (tiny step)']
            },
            '-': {
                'qt_key': Qt.Key_Minus,
                'slot': [self.scale_all],
                'parameter': [4 / 5],
                'description': ['Decrease Scale']
            },
            '+': {
                'qt_key': Qt.Key_Plus,
                'slot': [self.scale_all],
                'parameter': [5 / 4],
                'description': ['Increase Scale']
            },
            '=': {
                'qt_key': Qt.Key_Equal,
                'slot': [self.scale_all],
                'parameter': [5 / 4],
                'description': ['Increase Scale']
            },
            'a': {
                'qt_key': Qt.Key_A,
                'slot': [self._toggle_annotation_fig,
                         self._toggle_annotations],
                'modifier': [None, 'Shift'],
                'description': ['Toggle Annotation-Tool',
                                'Toggle Annotations visible']
            },
            'b': {
                'qt_key': Qt.Key_B,
                'slot': [self._toggle_butterfly],
                'description': ['Toggle Butterfly']
            },
            'd': {
                'qt_key': Qt.Key_D,
                'slot': [self._toggle_dc],
                'description': ['Toggle DC-Correction']
            },
            'e': {
                'qt_key': Qt.Key_E,
                'slot': [self._toggle_events],
                'description': ['Toggle Events visible']
            },
            'h': {
                'qt_key': Qt.Key_H,
                'slot': [self._toggle_epoch_histogram],
                'description': ['Toggle Epoch-Histogram']
            },
            'j': {
                'qt_key': Qt.Key_J,
                'slot': [self._toggle_proj_fig,
                         self._toggle_all_projs],
                'modifier': [None, 'Shift'],
                'description': ['Toggle Projection Figure',
                                'Toggle all projections']
            },
            'l': {
                'qt_key': Qt.Key_L,
                'slot': [self._toggle_antialiasing],
                'description': ['Toggle Antialiasing']
            },
            'o': {
                'qt_key': Qt.Key_O,
                'slot': [self._toggle_overview_bar],
                'description': ['Toggle Overview-Bar']
            },
            't': {
                'qt_key': Qt.Key_T,
                'slot': [self._toggle_time_format],
                'description': ['Toggle Time-Format']
            },
            's': {
                'qt_key': Qt.Key_S,
                'slot': [self._toggle_scalebars],
                'description': ['Toggle Scalebars']
            },
            'w': {
                'qt_key': Qt.Key_W,
                'slot': [self._toggle_whitening],
                'description': ['Toggle Whitening']
            },
            'x': {
                'qt_key': Qt.Key_X,
                'slot': [self._toggle_crosshair],
                'description': ['Toggle Crosshair']
            },
            'z': {
                'qt_key': Qt.Key_Z,
                'slot': [self._toggle_zenmode],
                'description': ['Toggle Zen-Mode']
            },
            '?': {
                'qt_key': Qt.Key_Question,
                'slot': [self._toggle_help_fig],
                'description': ['Show Help']
            },
            'f11': {
                'qt_key': Qt.Key_F11,
                'slot': [self._toggle_fullscreen],
                'description': ['Toggle Full-Screen']
            },
            'escape': {
                'qt_key': Qt.Key_Escape,
                'slot': [self.close],
                'description': ['Close']
            },
            # Just for testing
            'enter': {
                'qt_key': Qt.Key_Enter
            },
            ' ': {
                'qt_key': Qt.Key_Space
            }
        }

    def _get_scale_transform(self):
        transform = QTransform()
        transform.scale(1, self.mne.scale_factor)

        return transform

    def _update_yaxis_labels(self):
        self.mne.channel_axis.repaint()

    def _bad_ch_clicked(self, line):
        """Slot for bad channel click."""
        _, pick, marked_bad = self._toggle_bad_channel(line.range_idx)

        # Update line color
        line.isbad = not line.isbad
        line.update_color()

        # Update Channel-Axis
        self._update_yaxis_labels()

        # Update Overview-Bar
        self.mne.overview_bar.update_bad_channels()

        # update sensor color (if in selection mode)
        if self.mne.fig_selection is not None:
            self.mne.fig_selection._update_bad_sensors(pick, marked_bad)

    def _add_trace(self, ch_idx):
        trace = RawTraceItem(self.mne, ch_idx)

        # Apply scaling
        transform = self._get_scale_transform()
        trace.setTransform(transform)

        # Add Item early to have access to viewBox
        self.mne.plt.addItem(trace)
        self.mne.traces.append(trace)

        trace.sigClicked.connect(lambda tr, _: self._bad_ch_clicked(tr))

    def _remove_trace(self, trace):
        self.mne.plt.removeItem(trace)
        self.mne.traces.remove(trace)

    def _add_scalebars(self):
        """Add scalebars for all channel-types.
        (scene handles showing them in when in view
        range)
        """
        self.mne.scalebars.clear()
        # To keep order (np.unique sorts)
        ordered_types = self.mne.ch_types[self.mne.ch_order]
        unique_type_idxs = np.unique(ordered_types,
                                     return_index=True)[1]
        ch_types_ordered = [ordered_types[idx] for idx
                            in sorted(unique_type_idxs)]
        for ch_type in [ct for ct in ch_types_ordered
                        if ct != 'stim' and
                        ct in self.mne.scalings and
                        ct in getattr(self.mne, 'units', {}) and
                        ct in getattr(self.mne, 'unit_scalings', {})]:

            scale_bar = ScaleBar(self.mne, ch_type)
            self.mne.scalebars[ch_type] = scale_bar
            self.mne.plt.addItem(scale_bar)

            scale_bar_text = ScaleBarText(self.mne, ch_type)
            self.mne.scalebar_texts[ch_type] = scale_bar_text
            self.mne.plt.addItem(scale_bar_text)

    def _update_scalebar_x_positions(self):
        if self.mne.scalebars_visible:
            for scalebar in self.mne.scalebars.values():
                scalebar.update_x_position()

            for scalebar_text in self.mne.scalebar_texts.values():
                scalebar_text.update_x_position()

    def _update_scalebar_y_positions(self):
        if self.mne.scalebars_visible:
            for scalebar in self.mne.scalebars.values():
                scalebar.update_y_position()

            for scalebar_text in self.mne.scalebar_texts.values():
                scalebar_text.update_y_position()

    def _update_scalebar_values(self):
        for scalebar_text in self.mne.scalebar_texts.values():
            scalebar_text.update_value()

    def _set_scalebars_visible(self, visible):
        for scalebar in self.mne.scalebars.values():
            scalebar.setVisible(visible)

        for scalebar_text in self.mne.scalebar_texts.values():
            scalebar_text.setVisible(visible)

        self._update_scalebar_y_positions()

    def _toggle_scalebars(self):
        self.mne.scalebars_visible = not self.mne.scalebars_visible
        self._set_scalebars_visible(self.mne.scalebars_visible)

    def _overview_mode_changed(self, new_mode):
        self.mne.overview_mode = new_mode
        if self.mne.overview_mode == 'zscore':
            while self.mne.zscore_rgba is None:
                QApplication.processEvents()
        self.mne.overview_bar.set_background()

    def scale_all(self, step):
        """Scale all traces by multiplying with step."""
        self.mne.scale_factor *= step
        transform = self._get_scale_transform()

        # Scale Traces (by scaling the Item, not the data)
        for line in self.mne.traces:
            line.setTransform(transform)
            if self.mne.clipping is not None:
                line.update_data()

        # Update Scalebars
        self._update_scalebar_values()

    def hscroll(self, step):
        """Scroll horizontally by step."""
        if step == '+full':
            rel_step = self.mne.duration
        elif step == '-full':
            rel_step = - self.mne.duration
        else:
            rel_step = step * self.mne.duration / self.mne.scroll_sensitivity
        # Get current range and add step to it
        xmin, xmax = [i + rel_step for i in self.mne.viewbox.viewRange()[0]]

        if xmin < 0:
            xmin = 0
            xmax = xmin + self.mne.duration
        elif xmax > self.mne.xmax:
            xmax = self.mne.xmax
            xmin = xmax - self.mne.duration

        self.mne.plt.setXRange(xmin, xmax, padding=0)

    def vscroll(self, step):
        """Scroll vertically by step."""
        if self.mne.fig_selection is not None:
            if step == '+full':
                step = 1
            elif step == '-full':
                step = -1
            else:
                step = int(step)
            self.mne.fig_selection._scroll_selection(step)
        else:
            # Get current range and add step to it
            if step == '+full':
                step = self.mne.n_channels
            elif step == '-full':
                step = - self.mne.n_channels
            ymin, ymax = [i + step for i in self.mne.viewbox.viewRange()[1]]

            if ymin < 0:
                ymin = 0
                ymax = self.mne.n_channels + 1
            elif ymax > self.mne.ymax:
                ymax = self.mne.ymax
                ymin = ymax - self.mne.n_channels - 1

            self.mne.plt.setYRange(ymin, ymax, padding=0)

    def change_duration(self, step):
        """Change duration by step."""
        rel_step = self.mne.duration * step
        xmin, xmax = self.mne.viewbox.viewRange()[0]

        if self.mne.is_epochs:
            # use the length of one epoch as duration change
            min_dur = len(self.mne.inst.times) / self.mne.info['sfreq']
        else:
            # never show fewer than 3 samples
            min_dur = 3 * np.diff(self.mne.inst.times[:2])[0]

        xmax += rel_step

        if xmax - xmin < min_dur:
            xmax = xmin + min_dur

        if xmax > self.mne.xmax:
            diff = xmax - self.mne.xmax
            xmax = self.mne.xmax
            xmin -= diff

        if xmin < 0:
            xmin = 0

        self.mne.plt.setXRange(xmin, xmax, padding=0)

    def change_nchan(self, step):
        """Change number of channels by step."""
        if not self.mne.butterfly:
            if step == '+full':
                step = self.mne.n_channels
            elif step == '-full':
                step = - self.mne.n_channels
            ymin, ymax = self.mne.viewbox.viewRange()[1]
            ymax += step
            if ymax > self.mne.ymax:
                ymax = self.mne.ymax
                ymin -= step

            if ymin < 0:
                ymin = 0

            if ymax - ymin <= 2:
                ymax = ymin + 2

            self.mne.plt.setYRange(ymin, ymax, padding=0)

    def _remove_vline(self):
        if self.mne.vline:
            if self.mne.is_epochs:
                for vline in self.mne.vline:
                    self.mne.plt.removeItem(vline)
            else:
                self.mne.plt.removeItem(self.mne.vline)

        self.mne.vline = None
        self.mne.vline_visible = False

    def _add_vline(self, pos):
        # Remove vline if already shown
        self._remove_vline()

        self.mne.vline = VLine(pos, bounds=(0, self.mne.xmax))
        self.mne.plt.addItem(self.mne.vline)
        self.mne.vline_visible = True

    def _mouse_moved(self, pos):
        """Show Crosshair if enabled at mouse move."""
        if self.mne.crosshair_enabled:
            if self.mne.plt.sceneBoundingRect().contains(pos):
                mousePoint = self.mne.viewbox.mapSceneToView(pos)
                x, y = mousePoint.x(), mousePoint.y()
                if (0 <= x <= self.mne.xmax and
                        0 <= y <= self.mne.ymax):
                    if not self.mne.crosshair:
                        self.mne.crosshair = Crosshair()
                        self.mne.plt.addItem(self.mne.crosshair,
                                             ignoreBounds=True)

                    # Get ypos from trace
                    trace = [tr for tr in self.mne.traces if
                             tr.ypos - 0.5 < y < tr.ypos + 0.5]
                    if len(trace) == 1:
                        trace = trace[0]
                        idx = np.argmin(np.abs(trace.xData - x))
                        yshown = trace.get_ydata()[idx]

                        self.mne.crosshair.set_data(x, yshown)

                        yvalue = yshown - trace.ypos
                        # negative because plot is inverted for Y
                        scaler = -1 if self.mne.butterfly else -2
                        inv_norm = (scaler *
                                    self.mne.scalings[trace.ch_type] *
                                    self.mne.unit_scalings[trace.ch_type] /
                                    self.mne.scale_factor)
                        label = f'{_simplify_float(yvalue * inv_norm)} ' \
                                f'{self.mne.units[trace.ch_type]}'
                        self.statusBar().showMessage(f'x={x:.3f} s, '
                                                     f'y={label}')

    def _toggle_crosshair(self):
        self.mne.crosshair_enabled = not self.mne.crosshair_enabled
        if self.mne.crosshair:
            self.mne.plt.removeItem(self.mne.crosshair)
            self.mne.crosshair = None

    def _xrange_changed(self, _, xrange):
        # Update data
        self.mne.t_start = xrange[0]
        self.mne.duration = xrange[1] - xrange[0]
        self._redraw(update_data=True)

        # Update annotations
        self._update_annotations_xrange(xrange)

        # Update Events
        self._update_events_xrange(xrange)

        # Update Time-Bar
        self.mne.ax_hscroll.update_value(xrange[0])

        # Update Overview-Bar
        self.mne.overview_bar.update_viewrange()

        # Update Scalebars
        self._update_scalebar_x_positions()

    def _update_events_xrange(self, xrange):
        """Add or remove event-lines depending on view-range.

        This has proven to be more performant (and scalable)
        than adding all event-lines to plt(the Scene)
        and letting pyqtgraph/Qt handle it.
        """
        if self.mne.events_visible:
            for ev_line in self.mne.event_lines:
                if xrange[0] < ev_line.pos().x() < xrange[1]:
                    if ev_line not in self.mne.plt.items:
                        self.mne.plt.addItem(ev_line)
                else:
                    if ev_line in self.mne.plt.items:
                        self.mne.plt.removeItem(ev_line)

    def _update_annotations_xrange(self, xrange):
        """Add or remove annotation-regions depending on view-range.

        This has proven to be more performant (and scalable)
        than adding all annotations to plt(the Scene)
        and letting pyqtgraph/Qt handle it.
        """
        if self.mne.annotations_visible:
            for region in self.mne.regions:
                if self.mne.visible_annotations[region.description]:
                    rmin, rmax = region.getRegion()
                    xmin, xmax = xrange
                    comparisons = [rmin < xmin,
                                   rmin < xmax,
                                   rmax < xmin,
                                   rmax < xmax]
                    if all(comparisons) or not any(comparisons):
                        if region in self.mne.plt.items:
                            self.mne.plt.removeItem(region)
                            self.mne.plt.removeItem(region.label_item)
                    else:
                        if region not in self.mne.plt.items:
                            self.mne.plt.addItem(region)
                            self.mne.plt.addItem(region.label_item)

    def _yrange_changed(self, _, yrange):
        if not self.mne.butterfly:
            if not self.mne.fig_selection:
                # Update picks and data
                self.mne.ch_start = np.clip(round(yrange[0]), 0,
                                            len(self.mne.ch_order)
                                            - self.mne.n_channels)
                self.mne.n_channels = round(yrange[1] - yrange[0] - 1)
                self._update_picks()
                # Update Channel-Bar
                self.mne.ax_vscroll.update_value(self.mne.ch_start)
            self._update_data()

        # Update Overview-Bar
        self.mne.overview_bar.update_viewrange()

        # Update Scalebars
        self._update_scalebar_y_positions()

        off_traces = [tr for tr in self.mne.traces
                      if tr.ch_idx not in self.mne.picks]
        add_idxs = [p for p in self.mne.picks
                    if p not in [tr.ch_idx for tr in self.mne.traces]]

        # Update range_idx for traces which just shifted in y-position
        for trace in [tr for tr in self.mne.traces if tr not in off_traces]:
            trace.update_range_idx()

        # Update number of traces.
        trace_diff = len(self.mne.picks) - len(self.mne.traces)

        # Remove unnecessary traces.
        if trace_diff < 0:
            # Only remove from traces not in picks.
            remove_traces = off_traces[:abs(trace_diff)]
            for trace in remove_traces:
                self._remove_trace(trace)
                off_traces.remove(trace)

        # Add new traces if necessary.
        if trace_diff > 0:
            # Make copy to avoid skipping iteration.
            idxs_copy = add_idxs.copy()
            for aidx in idxs_copy[:trace_diff]:
                self._add_trace(aidx)
                add_idxs.remove(aidx)

        # Update data of traces outside of yrange (reuse remaining trace-items)
        for trace, ch_idx in zip(off_traces, add_idxs):
            trace.set_ch_idx(ch_idx)
            trace.update_color()
            trace.update_data()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # DATA HANDLING
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def _apply_downsampling(self):
        """
        Get ds-factor and apply ds with one of multiple methods.

        The methods are taken from PlotDataItem in pyqtgraph
        and adjusted to multi-channel data.
        """
        # Get Downsampling-Factor
        # Auto-Downsampling from pyqtgraph
        if self.mne.downsampling == 'auto':
            ds = 1
            if all([hasattr(self.mne, a) for a in ['viewbox', 'times']]):
                vb = self.mne.viewbox
                if vb is not None:
                    view_range = vb.viewRect()
                else:
                    view_range = None
                if view_range is not None and len(self.mne.times) > 1:
                    dx = float(self.mne.times[-1] - self.mne.times[0]) / (
                            len(self.mne.times) - 1)
                    if dx != 0.0:
                        x0 = view_range.left() / dx
                        x1 = view_range.right() / dx
                        width = vb.width()
                        if width != 0.0:
                            # Auto-Downsampling with 5 samples per pixel
                            ds = int(max(1, (x1 - x0) / (width * 5)))
        else:
            ds = self.mne.downsampling

        # Apply Downsampling
        if ds not in [None, 1]:
            times = self.mne.times
            data = self.mne.data
            n_ch = data.shape[0]

            if self.mne.ds_method == 'subsample':
                times = times[::ds]
                data = data[:, ::ds]

            elif self.mne.ds_method == 'mean':
                n = len(times) // ds
                # start of x-values
                # try to select a somewhat centered point
                stx = ds // 2
                times = times[stx:stx + n * ds:ds]
                rs_data = data[:, :n * ds].reshape(n_ch, n, ds)
                data = rs_data.mean(axis=2)

            elif self.mne.ds_method == 'peak':
                n = len(times) // ds
                # start of x-values
                # try to select a somewhat centered point
                stx = ds // 2

                x1 = np.empty((n, 2))
                x1[:] = times[stx:stx + n * ds:ds, np.newaxis]
                times = x1.reshape(n * 2)

                y1 = np.empty((n_ch, n, 2))
                y2 = data[:, :n * ds].reshape((n_ch, n, ds))
                y1[:, :, 0] = y2.max(axis=2)
                y1[:, :, 1] = y2.min(axis=2)
                data = y1.reshape((n_ch, n * 2))

            self.mne.times, self.mne.data = times, data

    def _show_process(self, message):
        if self.mne.load_progressbar.isVisible():
            self.mne.load_progressbar.hide()
            self.mne.load_prog_label.hide()
        self.statusBar().showMessage(message)

    def _precompute_finished(self):
        self.statusBar().showMessage('Loading Finished', 5)
        self.mne.data_precomputed = True

        if self.mne.overview_mode == 'zscore':
            # Show loaded overview image
            self.mne.overview_bar.set_background()

    def _init_precompute(self):
        # Remove previously loaded data
        self.mne.data_precomputed = False
        if all([hasattr(self.mne, st)
                for st in ['global_data', 'global_times']]):
            del self.mne.global_data, self.mne.global_times
        gc.collect()

        if self.mne.precompute == 'auto':
            self.mne.enable_precompute = self._check_space_for_precompute()
        elif isinstance(self.mne.precompute, bool):
            self.mne.enable_precompute = self.mne.precompute

        if self.mne.enable_precompute:
            # Start precompute thread
            self.mne.load_progressbar.show()
            self.mne.load_prog_label.show()
            load_runner = LoadRunner(self)
            load_runner.sigs.loadProgress.connect(self.mne.
                                                  load_progressbar.setValue)
            load_runner.sigs.processText.connect(self._show_process)
            load_runner.sigs.loadingFinished.connect(self._precompute_finished)
            QThreadPool.globalInstance().start(load_runner)

    def _check_space_for_precompute(self):
        try:
            import psutil
        except ImportError:
            logger.info('Free RAM space could not be determined because'
                        '"psutil" is not installed. '
                        'Setting precompute to False.')
            return False
        else:
            if self.mne.inst.filenames[0]:
                # Get disk-space of raw-file(s)
                disk_space = 0
                for fn in self.mne.inst.filenames:
                    disk_space += getsize(fn)

                # Determine expected RAM space based on orig_format
                fmt_multipliers = {'double': 1,
                                   'single': 2,
                                   'int': 2,
                                   'short': 4}

                fmt = self.mne.inst.orig_format
                # Apply size change to 64-bit float in memory
                # (* 2 because when loading data will be loaded into a copy
                # of self.mne.inst._data to apply processing.
                expected_ram = disk_space * fmt_multipliers[fmt] * 2
            else:
                expected_ram = sys.getsizeof(self.mne.inst._data)

            # Get available RAM
            free_ram = psutil.virtual_memory().free

            expected_ram_str = sizeof_fmt(expected_ram)
            free_ram_str = sizeof_fmt(free_ram)
            left_ram_str = sizeof_fmt(free_ram - expected_ram)

            if expected_ram < free_ram:
                logger.debug('The data precomputed for visualization takes '
                             f'{expected_ram_str} with {left_ram_str} of '
                             f'RAM left.')
                return True
            else:
                logger.debug(f'The precomputed data with {expected_ram_str} '
                             f'will surpass your current {free_ram_str} '
                             f'of free RAM.\n'
                             'Thus precompute will be set to False.\n'
                             '(If you want to precompute nevertheless, '
                             'then set precompute to True instead of "auto")')
                return False

    def _process_data(self, data, start, stop, picks,
                      signals=None):
        data = super()._process_data(data, start, stop, picks, signals)

        # Invert Data to be displayed from top on inverted Y-Axis
        data *= -1

        return data

    def _update_data(self):
        if self.mne.data_precomputed:
            # get start/stop-samples
            start, stop = self._get_start_stop()
            self.mne.times = self.mne.global_times[start:stop]
            self.mne.data = self.mne.global_data[:, start:stop]

            # remove DC locally
            if self.mne.remove_dc:
                self.mne.data = self.mne.data - \
                                self.mne.data.mean(axis=1, keepdims=True)

        else:
            # While data is not precomputed get data only from shown range and
            # process only those.
            super()._update_data()

        # Initialize decim
        self.mne.decim_data = np.ones_like(self.mne.picks)
        data_picks_mask = np.in1d(self.mne.picks, self.mne.picks_data)
        self.mne.decim_data[data_picks_mask] = self.mne.decim

        # Get decim_times
        if self.mne.decim != 1:
            # decim can vary by channel type,
            # so compute different `times` vectors.
            self.mne.decim_times = {decim_value: self.mne.times[::decim_value]
                                    + self.mne.first_time for
                                    decim_value in set(self.mne.decim_data)}

        # Apply clipping
        if self.mne.clipping == 'clamp':
            self.mne.data = np.clip(self.mne.data, -0.5, 0.5)
        elif self.mne.clipping is not None:
            self.mne.data = self.mne.data.copy()
            self.mne.data[abs(self.mne.data * self.mne.scale_factor)
                          > self.mne.clipping] = np.nan

        # Apply Downsampling (if enabled)
        self._apply_downsampling()

    def _get_zscore(self, data):
        # Reshape data to reasonable size for display
        max_pixel_width = QApplication.desktop().screenGeometry().width()
        collapse_by = data.shape[1] // max_pixel_width
        data = data[:, :max_pixel_width * collapse_by]
        if collapse_by > 0:
            data = data.reshape(data.shape[0], max_pixel_width, collapse_by)
            data = data.mean(axis=2)
        z = zscore(data, axis=1)
        if z.size > 0:
            zmin = np.min(z, axis=1)
            zmax = np.max(z, axis=1)

            # Convert into RGBA
            zrgba = np.empty((*z.shape, 4))
            for row_idx, row in enumerate(z):
                for col_idx, value in enumerate(row):
                    if math.isnan(value):
                        value = 0
                    if value == 0:
                        rgba = [0, 0, 0, 0]
                    elif value < 0:
                        alpha = int(255 * value / abs(zmin[row_idx]))
                        rgba = [0, 0, 255, alpha]
                    else:
                        alpha = int(255 * value / zmax[row_idx])
                        rgba = [255, 0, 0, alpha]

                    zrgba[row_idx, col_idx] = rgba

            zrgba = np.require(zrgba, np.uint8, 'C')

            self.mne.zscore_rgba = zrgba

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # ANNOTATIONS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def _add_region(self, plot_onset, duration, description, region=None):
        if not region:
            region = AnnotRegion(self.mne, description=description,
                                 values=(plot_onset, plot_onset + duration))
        if (any([self.mne.t_start < v < self.mne.t_start + self.mne.duration
                for v in [plot_onset, plot_onset + duration]]) and
                region not in self.mne.plt.items):
            self.mne.plt.addItem(region)
            self.mne.plt.addItem(region.label_item)
        region.regionChangeFinished.connect(self._region_changed)
        region.gotSelected.connect(self._region_selected)
        region.removeRequested.connect(self._remove_region)
        self.mne.viewbox.sigYRangeChanged.connect(
            region.update_label_pos)
        self.mne.regions.append(region)

        region.update_label_pos()

    def _remove_region(self, region, from_annot=True):
        # Remove from shown regions
        if region.label_item in self.mne.viewbox.addedItems:
            self.mne.viewbox.removeItem(region.label_item)
        if region in self.mne.plt.items:
            self.mne.plt.removeItem(region)

        # Remove from all regions
        if region in self.mne.regions:
            self.mne.regions.remove(region)

        # Reset selected region
        if region == self.mne.selected_region:
            self.mne.selected_region = None

        # Remove from annotations
        if from_annot:
            idx = self._get_onset_idx(region.getRegion()[0])
            self.mne.inst.annotations.delete(idx)

        # Update Overview-Bar
        self.mne.overview_bar.update_annotations()

    def _region_selected(self, region):
        old_region = self.mne.selected_region
        # Remove selected-status from old region
        if old_region and old_region != region:
            old_region.select(False)
        self.mne.selected_region = region
        self.mne.fig_annotation.update_values(region)

    def _get_onset_idx(self, plot_onset):
        onset = _sync_onset(self.mne.inst, plot_onset, inverse=True)
        idx = np.where(self.mne.inst.annotations.onset == onset)[0][0]
        return idx

    def _region_changed(self, region):
        rgn = region.getRegion()
        region.select(True)
        idx = self._get_onset_idx(region.old_onset)

        # Update Spinboxes of Annot-Dock
        self.mne.fig_annotation.update_values(region)

        # Change annotations
        self.mne.inst.annotations.onset[idx] = _sync_onset(self.mne.inst,
                                                           rgn[0],
                                                           inverse=True)
        self.mne.inst.annotations.duration[idx] = rgn[1] - rgn[0]

        # Update overview-bar
        self.mne.overview_bar.update_annotations()

    def _draw_annotations(self):
        # All regions are constantly added to the Scene and handled by Qt
        # which is faster than handling adding/removing in Python.
        pass

    def _change_annot_mode(self):
        if not self.mne.annotation_mode:
            # Reset Widgets in Annotation-Figure
            self.mne.fig_annotation.reset()

        # Show Annotation-Dock if activated.
        self.mne.fig_annotation.setVisible(self.mne.annotation_mode)

        # Make Regions movable if activated and move into foreground
        for region in self.mne.regions:
            region.setMovable(self.mne.annotation_mode)
            if self.mne.annotation_mode:
                region.setZValue(2)
            else:
                region.setZValue(0)

        # Add/Remove selection-rectangle.
        if self.mne.selected_region:
            self.mne.selected_region.select(self.mne.annotation_mode)

    def _toggle_annotation_fig(self):
        self.mne.annotation_mode = not self.mne.annotation_mode
        self._change_annot_mode()

    def _update_regions_visible(self):
        for region in self.mne.regions:
            region.update_visible(
                self.mne.visible_annotations[region.description])
        self.mne.overview_bar.update_annotations()

    def _set_annotations_visible(self, visible):
        for descr in self.mne.visible_annotations:
            self.mne.visible_annotations[descr] = visible
        self._update_regions_visible()

        # Update Plot
        if visible:
            self._update_annotations_xrange((self.mne.t_start,
                                             self.mne.t_start +
                                             self.mne.duration))
        else:
            for region in [r for r in self.mne.regions
                           if r in self.mne.plt.items]:
                self.mne.plt.removeItem(region)
                self.mne.plt.removeItem(region.label_item)

    def _toggle_annotations(self):
        self.mne.annotations_visible = not self.mne.annotations_visible
        self._set_annotations_visible(self.mne.annotations_visible)

    def _apply_update_projectors(self, toggle_all=False):
        if toggle_all:
            on = self.mne.projs_on
            applied = self.mne.projs_active
            value = False if all(on) else True
            new_state = np.full_like(on, value)
            # Always activate applied projections
            new_state[applied] = True
            self.mne.projs_on = new_state
        self._update_projector()
        # If data was precomputed it needs to be precomputed again.
        self._init_precompute()
        self._redraw()

    def _toggle_proj_fig(self):
        if self.mne.fig_proj is None:
            ProjDialog(self, name='fig_proj')
        else:
            self.mne.fig_proj.close()

    def _toggle_all_projs(self):
        if self.mne.fig_proj is None:
            self._apply_update_projectors(toggle_all=True)
        else:
            self.mne.fig_proj.toggle_all()

    def _toggle_whitening(self):
        super()._toggle_whitening()
        # If data was precomputed it needs to be precomputed again.
        self._init_precompute()
        self._redraw()

    def _toggle_settings_fig(self):
        if self.mne.fig_settings is None:
            SettingsDialog(self, name='fig_settings')
        else:
            self.mne.fig_help.close()
            self.mne.fig_help = None

    def _toggle_help_fig(self):
        if self.mne.fig_help is None:
            HelpDialog(self, name='fig_help')
        else:
            self.mne.fig_help.close()
            self.mne.fig_help = None

    def _set_butterfly(self, butterfly):
        self.mne.butterfly = butterfly
        self._update_picks()
        self._update_data()

        if butterfly and self.mne.fig_selection is not None:
            self.mne.selection_ypos_dict.clear()
            selections_dict = self._make_butterfly_selections_dict()
            for idx, picks in enumerate(selections_dict.values()):
                for pick in picks:
                    self.mne.selection_ypos_dict[pick] = idx + 1
            ymax = len(selections_dict) + 1
            self.mne.plt.setLimits(yMax=ymax)
            self.mne.plt.setYRange(0, ymax, padding=0)
        elif butterfly:
            ymax = len(self.mne.butterfly_type_order) + 1
            self.mne.plt.setLimits(yMax=ymax)
            self.mne.plt.setYRange(0, ymax, padding=0)
        else:
            self.mne.plt.setLimits(yMax=self.mne.ymax)
            self.mne.plt.setYRange(self.mne.ch_start,
                                   self.mne.ch_start + self.mne.n_channels + 1,
                                   padding=0)

        if self.mne.fig_selection is not None:
            # Update Selection-Dialog
            self.mne.fig_selection._style_butterfly()

        # Set vertical scrollbar visible
        self.mne.ax_vscroll.setVisible(not butterfly or
                                       self.mne.fig_selection is not None)

        # update overview-bar
        self.mne.overview_bar.update_viewrange()

        # update ypos and color for butterfly-mode
        for trace in self.mne.traces:
            trace.update_ypos()
            trace.update_color()

        self._draw_traces()

    def _toggle_butterfly(self):
        self._set_butterfly(not self.mne.butterfly)

    def _toggle_dc(self):
        self.mne.remove_dc = not self.mne.remove_dc
        self._redraw()

    def _toggle_epoch_histogram(self):
        fig = self._create_epoch_histogram()
        self._get_dlg_from_mpl(fig)

    def _set_events_visible(self, visible):
        for event_line in self.mne.event_lines:
            event_line.setVisible(visible)

        # Update Plot
        if visible:
            self._update_events_xrange((self.mne.t_start,
                                        self.mne.t_start +
                                        self.mne.duration))
        else:
            for event_line in [evl for evl in self.mne.event_lines
                               if evl in self.mne.plt.items]:
                self.mne.plt.removeItem(event_line)
        self.mne.overview_bar.update_events()

    def _toggle_events(self):
        if self.mne.event_nums is not None:
            self.mne.events_visible = not self.mne.events_visible
            self._set_events_visible(self.mne.events_visible)

    def _toggle_time_format(self):
        if self.mne.time_format == 'float':
            self.mne.time_format = 'clock'
            self.mne.time_axis.setLabel(text='Time')
        else:
            self.mne.time_format = 'float'
            self.mne.time_axis.setLabel(text='Time', units='s')
        self._update_yaxis_labels()

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _toggle_antialiasing(self):
        self.mne.antialiasing = not self.mne.antialiasing
        self._redraw()

    def _toggle_overview_bar(self):
        self.mne.show_overview_bar = not self.mne.show_overview_bar
        self.mne.overview_bar.setVisible(self.mne.show_overview_bar)

    def _toggle_zenmode(self):
        self.mne.scrollbars_visible = not self.mne.scrollbars_visible
        for bar in [self.mne.ax_hscroll, self.mne.ax_vscroll]:
            bar.setVisible(self.mne.scrollbars_visible)
        self.mne.toolbar.setVisible(self.mne.scrollbars_visible)

    def _new_child_figure(self, fig_name, window_title, **kwargs):
        from matplotlib.figure import Figure
        fig = Figure(**kwargs)
        # Pass window title and fig_name on
        if fig_name is not None:
            fig.fig_name = fig_name
        if window_title is not None:
            fig.title = window_title
        return fig

    def _get_widget_from_mpl(self, fig):
        canvas = FigureCanvasQTAgg(fig)
        canvas.setFocusPolicy(Qt.StrongFocus | Qt.WheelFocus)
        canvas.setFocus()
        # Pass window title and fig_name on
        if hasattr(fig, 'fig_name'):
            canvas.fig_name = fig.fig_name
        if hasattr(fig, 'title'):
            canvas.title = fig.title

        return canvas

    def _get_dlg_from_mpl(self, fig):
        canvas = self._get_widget_from_mpl(fig)
        # Pass window title and fig_name on
        if hasattr(canvas, 'fig_name'):
            name = canvas.fig_name
        else:
            name = None
        if hasattr(canvas, 'title'):
            title = canvas.title
        else:
            title = None
        dlg = _BaseDialog(self, widget=canvas, title=title, name=name)
        dlg.show()

    def _create_ch_context_fig(self, idx):
        fig = super()._create_ch_context_fig(idx)
        if fig is not None:
            self._get_dlg_from_mpl(fig)

    def _update_trace_offsets(self):
        pass

    def _create_selection_fig(self):
        SelectionDialog(self)

    def keyPressEvent(self, event):
        """Customize key press events."""
        # On MacOs additionally KeypadModifier is set when arrow-keys
        # are pressed.
        # On Unix GroupSwitchModifier is set when ctrl is pressed.
        # To preserve cross-platform consistency the following comparison
        # of the modifier-values is done.
        # modifiers need to be exclusive
        modifiers = {
            'Ctrl': '4' in hex(int(event.modifiers())),
            'Shift': int(event.modifiers()) == 33554432
        }
        for key_name in self.mne.keyboard_shortcuts:
            key_dict = self.mne.keyboard_shortcuts[key_name]
            if key_dict['qt_key'] == event.key() and 'slot' in key_dict:

                mod_idx = 0
                # Get modifier
                if 'modifier' in key_dict:
                    mods = [modifiers[mod] for mod in modifiers]
                    if any(mods):
                        # No multiple modifiers supported yet
                        mod = [mod for mod in modifiers if modifiers[mod]][0]
                        if mod in key_dict['modifier']:
                            mod_idx = key_dict['modifier'].index(mod)

                slot_idx = mod_idx if mod_idx < len(key_dict['slot']) else 0
                slot = key_dict['slot'][slot_idx]

                if 'parameter' in key_dict:
                    param_idx = (mod_idx if mod_idx <
                                 len(key_dict['parameter']) else 0)
                    slot(key_dict['parameter'][param_idx])
                else:
                    slot()

                break

    def _draw_traces(self):
        # Update data in traces (=drawing traces)
        for trace in self.mne.traces:
            # Update data
            trace.update_data()

    def _get_size(self):
        inch_width = self.width() / self.logicalDpiX()
        inch_height = self.height() / self.logicalDpiY()

        return inch_width, inch_height

    def _fake_keypress(self, key, fig=None):
        fig = fig or self

        if key.isupper():
            key = key.lower()
            modifier = Qt.ShiftModifier
        else:
            modifier = Qt.NoModifier

        # Use pytest-qt's exception-hook
        with capture_exceptions() as exceptions:
            QTest.keyPress(fig, self.mne.keyboard_shortcuts[key]['qt_key'],
                           modifier)

        for exc in exceptions:
            raise RuntimeError(f'There as been an {exc[0]} inside the Qt '
                               f'event loop (look above for traceback).')

    def _fake_click(self, point, add_points=None, fig=None, ax=None,
                    xform='ax', button=1, kind='press'):
        add_points = add_points or list()
        # Wait until Window is fully shown.
        QTest.qWaitForWindowExposed(self)
        # Scene-Dimensions still seem to change to final state when waiting
        # for a short time.
        QTest.qWait(10)

        # Qt: right-button=2, matplotlib: right-button=3
        if button == 1:
            button = Qt.LeftButton
        else:
            button = Qt.RightButton

        # For Qt, fig or ax both would be the widget to test interaction on.
        # If View
        fig = ax or fig or self.mne.view

        if xform == 'ax':
            # For Qt, the equivalent of matplotlibs transAxes
            # would be a transformation to View Coordinates.
            # But for the View top-left is (0, 0) and bottom-right is
            # (view-width, view-height).
            view_width = fig.width()
            view_height = fig.height()
            x = view_width * point[0]
            y = view_height * (1 - point[1])
            point = Point(x, y)
            for idx, apoint in enumerate(add_points):
                x2 = view_width * apoint[0]
                y2 = view_height * (1 - apoint[1])
                add_points[idx] = Point(x2, y2)

        elif xform == 'data':
            # For Qt, the equivalent of matplotlibs transData
            # would be a transformation to
            # the coordinate system of the ViewBox.
            # This only works on the View (self.mne.view)
            fig = self.mne.view
            point = self.mne.viewbox.mapViewToScene(Point(*point))
            for idx, apoint in enumerate(add_points):
                add_points[idx] = self.mne.viewbox.mapViewToScene(
                    Point(*apoint))

        elif xform == 'none' or xform is None:
            if isinstance(point, (tuple, list)):
                point = Point(*point)
            else:
                point = Point(point)
            for idx, apoint in enumerate(add_points):
                if isinstance(apoint, (tuple, list)):
                    add_points[idx] = Point(*apoint)
                else:
                    add_points[idx] = Point(apoint)

        # Use pytest-qt's exception-hook
        with capture_exceptions() as exceptions:
            widget = fig.viewport() if isinstance(fig, QGraphicsView) else fig
            if kind == 'press':
                # always click because most interactivity comes form
                # mouseClickEvent from pyqtgraph (just press doesn't suffice
                # here).
                _mouseClick(widget=widget, pos=point, button=button)
            elif kind == 'release':
                _mouseRelease(widget=widget, pos=point, button=button)
            elif kind == 'motion':
                _mouseMove(widget=widget, pos=point, buttons=button)
            elif kind == 'drag':
                _mouseDrag(widget=widget, positions=[point] + add_points,
                           button=button)

        for exc in exceptions:
            raise RuntimeError(f'There as been an {exc[0]} inside the Qt '
                               f'event loop (look above for traceback).')

        # Waiting some time for events to be processed.
        QTest.qWait(50)

    def _fake_scroll(self, x, y, step, fig=None):
        # QTest doesn't support simulating scrolling-wheel
        self.vscroll(step)

    def _click_ch_name(self, ch_index, button):
        if not self.mne.butterfly:
            ch_name = self.mne.ch_names[self.mne.picks[ch_index]]
            xrange, yrange = self.mne.channel_axis.ch_texts[ch_name]
            x = np.mean(xrange)
            y = np.mean(yrange)

            self._fake_click((x, y), fig=self.mne.view, button=button,
                             xform='none')

    def _resize_by_factor(self, factor):
        pass

    def _get_ticklabels(self, orientation):
        if orientation == 'x':
            ax = self.mne.time_axis
        else:
            ax = self.mne.channel_axis

        return list(ax.get_labels())

    def _get_scale_bar_texts(self):
        return tuple(t.toPlainText() for t in self.mne.scalebar_texts.values())

    def _close_event(self, fig=None):
        """Force calling of the MPL figure close event."""
        fig = fig or self
        if hasattr(fig, 'canvas'):
            try:
                fig.canvas.close_event()
            except ValueError:  # old mpl with Qt
                pass  # pragma: no cover
        else:
            fig.close()

    def closeEvent(self, event):
        """Customize close event."""
        event.accept()
        if hasattr(self, 'mne'):
            self._close(event)
        self.gotClosed.emit()


def _get_n_figs():
    # Wait for a short time to let the Qt-loop clean up
    QTest.qWait(100)
    return len([window for window in QApplication.topLevelWindows()
                if window.isVisible()])


def _close_all():
    if len(QApplication.topLevelWindows()) > 0:
        QApplication.closeAllWindows()


# mouse testing functions adapted from pyqtgraph
# (pyqtgraph.tests.ui_testing.py)
def _mousePress(widget, pos, button, modifier=None):
    if modifier is None:
        modifier = Qt.KeyboardModifier.NoModifier
    event = QMouseEvent(QEvent.Type.MouseButtonPress, pos, button,
                        Qt.MouseButton.NoButton, modifier)
    QApplication.sendEvent(widget, event)


def _mouseRelease(widget, pos, button, modifier=None):
    if modifier is None:
        modifier = Qt.KeyboardModifier.NoModifier
    event = QMouseEvent(QEvent.Type.MouseButtonRelease, pos,
                        button, Qt.MouseButton.NoButton, modifier)
    QApplication.sendEvent(widget, event)


def _mouseMove(widget, pos, buttons=None, modifier=None):
    if buttons is None:
        buttons = Qt.MouseButton.NoButton
    if modifier is None:
        modifier = Qt.KeyboardModifier.NoModifier
    event = QMouseEvent(QEvent.Type.MouseMove, pos,
                        Qt.MouseButton.NoButton, buttons, modifier)
    QApplication.sendEvent(widget, event)


def _mouseClick(widget, pos, button, modifier=None):
    _mouseMove(widget, pos)
    _mousePress(widget, pos, button, modifier)
    _mouseRelease(widget, pos, button, modifier)


def _mouseDrag(widget, positions, button, modifier=None):
    _mouseMove(widget, positions[0])
    _mousePress(widget, positions[0], button, modifier)
    # Delay for 10 ms for drag to be recognized.
    QTest.qWait(10)
    for pos in positions[1:]:
        _mouseMove(widget, pos, button, modifier)
    _mouseRelease(widget, positions[-1], button, modifier)


def _init_browser(**kwargs):
    setConfigOption('enableExperimental', True)

    app = mkQApp()
    _init_qt_resources()
    kind = 'bigsur-' if platform.mac_ver()[0] >= '10.16' else ''
    app.setWindowIcon(QIcon(f":/mne-{kind}icon.png"))
    browser = PyQtGraphBrowser(**kwargs)

    return browser
