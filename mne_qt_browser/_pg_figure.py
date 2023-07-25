# -*- coding: utf-8 -*-
"""Base classes and functions for 2D browser backends."""

# Author: Martin Schulz <dev@earthman-music.de>
#
# License: BSD-3-Clause

import datetime
import functools
import gc
import math
import platform
import sys
import weakref
from pathlib import Path
from ast import literal_eval
from collections import OrderedDict
from copy import copy
from functools import partial
from os.path import getsize

import numpy as np
from qtpy.QtCore import (QEvent, QThread, Qt, Signal, QRectF, QLineF,
                         QPointF, QPoint, QSettings, QSignalBlocker)
from qtpy.QtGui import (QFont, QIcon, QPixmap, QTransform, QGuiApplication,
                        QMouseEvent, QImage, QPainter, QPainterPath, QColor)
from qtpy.QtTest import QTest
from qtpy.QtWidgets import (QAction, QColorDialog, QComboBox, QDialog,
                            QDockWidget, QDoubleSpinBox, QFormLayout,
                            QGridLayout, QHBoxLayout, QInputDialog,
                            QLabel, QMainWindow, QMessageBox, QToolButton,
                            QPushButton, QScrollBar, QWidget, QMenu,
                            QStyleOptionSlider, QStyle, QActionGroup,
                            QApplication, QGraphicsView, QProgressBar,
                            QVBoxLayout, QLineEdit, QCheckBox, QScrollArea,
                            QGraphicsLineItem, QGraphicsScene, QTextEdit,
                            QSizePolicy, QSpinBox, QSlider, QWidgetAction,
                            QRadioButton)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.colors import to_rgba_array
from pyqtgraph import (AxisItem, GraphicsView, InfLineLabel, InfiniteLine,
                       LinearRegionItem, PlotCurveItem, PlotItem,
                       Point, TextItem, ViewBox, mkBrush,
                       mkPen, setConfigOption, mkColor)
from scipy.stats import zscore
from colorspacious import cspace_convert

import scooby
from mne.viz import plot_sensors
from mne.viz._figure import BrowserBase
from mne.viz.utils import _simplify_float, _merge_annotations, _figure_agg
from mne.annotations import _sync_onset
from mne.io.pick import (_DATA_CH_TYPES_ORDER_DEFAULT,
                         channel_indices_by_type, _DATA_CH_TYPES_SPLIT)
from mne.utils import (_to_rgb, logger, sizeof_fmt, warn, get_config,
                       _check_option)

from . import _browser_instances
from ._fixes import capture_exceptions, _qt_raise_window, _init_mne_qtapp

name = 'pyqtgraph'

# MNE's butterfly plots traditionally default to the channel ordering of
# mag, grad, ..., which is inconsistent with the order in non-butterfly mode
# and hence doesn't match the order in the overview bar either. So we swap
# grads and mags here.
DATA_CH_TYPES_ORDER = ('grad', 'mag', *_DATA_CH_TYPES_ORDER_DEFAULT[2:])


# Mostly chosen manually from
# https://matplotlib.org/3.1.0/gallery/color/named_colors.html
_dark_dict = {
    # 'w' (bgcolor)
    (255, 255, 255): (30, 30, 30),  # safari's centered info panel background
    # 'k' (eeg, eog, emg, misc, stim, resp, chpi, exci, ias, syst, dipole, gof,
    #      bio, ecog, fnirs_*, csd, whitened)
    (0, 0, 0): (255, 255, 255),   # 'w'
    # 'darkblue' (mag)
    (0, 0, 139): (173, 216, 230),  # 'lightblue'
    # 'b' (grad, hbr)
    (0, 0, 255): (100, 149, 237),  # 'cornflowerblue'
    # 'steelblue' (ref_meg)
    (70, 130, 180): (176, 196, 222),  # 'lightsteelblue'
    # 'm' (ecg)
    (191, 0, 191): (238, 130, 238),  # 'violet'
    # 'saddlebrown' (seeg)
    (139, 69, 19): (244, 164, 96),  # 'sandybrown'
    # 'seagreen' (dbs)
    (46, 139,  87): (32, 178, 170),  # 'lightseagreen'
    # '#AA3377' (hbo), closest to 'mediumvioletred'
    (170, 51, 119): (255, 105, 180),  # 'hotpink'
    # 'lightgray' (bad_color)
    (211, 211, 211): (105, 105, 105),  # 'dimgray'
    # 'cyan' (event_color)
    (0, 255, 255): (0, 139, 139),  # 'darkcyan'
}


def _get_color(color_spec, invert=False):
    """Wrap mkColor to accept all possible matplotlib color-specifiers."""
    if isinstance(color_spec, np.ndarray):
        color_spec = tuple(color_spec)
    # We have to pass to QColor here to make a copy because we should be able
    # to .setAlpha(...) etc. on it and this would otherwise affect the cache.
    return QColor(_get_color_cached(color_spec=color_spec, invert=invert))


@functools.lru_cache(maxsize=100)
def _get_color_cached(*, color_spec, invert):
    orig_spec = color_spec
    try:
        # Convert matplotlib color-names if possible
        color_spec = _to_rgb(color_spec, alpha=True)
    except ValueError:
        pass

    # Convert tuples of floats from 0-1 to 0-255 for pyqtgraph
    if (isinstance(color_spec, tuple) and
            all([i <= 1 for i in color_spec])):
        color_spec = tuple([int(i * 255) for i in color_spec])

    try:
        color = mkColor(color_spec)
    except ValueError:
        raise ValueError(f'"{color_spec}" is not a valid matplotlib '
                         f'color-specifier!') from None
    if invert:
        # First see if the color is in our inversion dictionary
        key = color.getRgb()
        assert len(key) == 4
        if key[:3] in _dark_dict:
            color.setRgb(*(_dark_dict[key[:3]] + key[-1:]))
        else:
            logger.debug(f'Missed {key} from {orig_spec}')
            rgba = np.array(color.getRgbF())
            lab = cspace_convert(rgba[:3], 'sRGB1', 'CIELab')
            lab[0] = 100. - lab[0]
            rgba[:3] = np.clip(cspace_convert(lab, 'CIELab', 'sRGB1'), 0, 1)
            color.setRgbF(*rgba)

    return color


def propagate_to_children(method):  # noqa: D103
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        propagate = kwargs.pop('propagate', True)
        result = method(*args, **kwargs)
        if args[0].mne.is_epochs and propagate:
            # parent always goes first
            if hasattr(args[0], 'child_traces'):
                for child_trace in args[0].child_traces:
                    getattr(child_trace, method.__name__)(*args[1:], **kwargs)
        return result

    return wrapper


def _safe_splash(meth):
    @functools.wraps(meth)
    def func(self, *args, **kwargs):
        try:
            meth(self, *args, **kwargs)
        finally:
            try:
                self.mne.splash.close()
            except Exception:
                pass
            finally:
                try:
                    del self.mne.splash
                except Exception:
                    pass
    return func


class DataTrace(PlotCurveItem):
    """Graphics-Object for single data trace."""

    def __init__(self, main, ch_idx, child_idx=None, parent_trace=None):
        super().__init__()
        self.weakmain = weakref.ref(main)
        self.mne = main.mne
        del main

        # Set clickable with small area around trace to make clicking easier.
        self.setClickable(True, 12)

        # Set default z-value to 1 to be before other items in scene
        self.setZValue(1)

        # General attributes
        # The ch_idx is the index of the channel represented by this trace
        # in the channel-order from the unchanged instance (which also picks
        # refer to).
        self.ch_idx = None
        # The range_idx is the index of the channel represented by this trace
        # in the shown range.
        self.range_idx = None
        # The order_idx is the index of the channel represented by this trace
        # in the channel-order (defined e.g. by group_by).
        self.order_idx = None
        # Name of the channel the trace represents.
        self.ch_name = None
        # Indicates if trace is bad.
        self.isbad = None
        # Channel-type of trace.
        self.ch_type = None
        # Color-specifier (all possible matplotlib color formats)
        self.color = None

        # Attributes for epochs-mode
        # Index of child if child.
        self.child_idx = child_idx
        # Reference to parent if child.
        self.parent_trace = parent_trace

        # Only for parent traces
        if self.parent_trace is None:
            # Add to main trace list
            self.mne.traces.append(self)
            # References to children
            self.child_traces = list()
            # Colors of trace in viewrange
            self.trace_colors = None

        # set attributes
        self.set_ch_idx(ch_idx)
        self.update_color()
        self.update_scale()
        # Avoid calling self.update_data() twice on initialization
        # (because of update_scale()).
        if self.mne.clipping is None:
            self.update_data()

        # Add to main plot
        self.mne.plt.addItem(self)

    @propagate_to_children
    def remove(self):  # noqa: D102
        self.mne.plt.removeItem(self)
        # Only for parent trace
        if self.parent_trace is None:
            self.mne.traces.remove(self)
        self.deleteLater()

    @propagate_to_children
    def update_color(self):
        """Update the color of the trace."""
        # Epochs
        if self.mne.is_epochs:
            # Add child traces if shown trace needs to have multiple colors
            # (PlotCurveItem only supports one color per object).
            # There always as many color-specific traces added depending
            # on the whole time range of the instance regardless of the
            # currently visible time range (to avoid checking for new colors
            # while scrolling horizontally).

            # Only for parent trace
            if hasattr(self, 'child_traces'):
                self.trace_colors = np.unique(
                        self.mne.epoch_color_ref[self.ch_idx], axis=0)
                n_childs = len(self.child_traces)
                trace_diff = len(self.trace_colors) - n_childs - 1
                # Add child traces if necessary
                if trace_diff > 0:
                    for cix in range(n_childs, n_childs + trace_diff):
                        child = DataTrace(self.weakmain(), self.ch_idx,
                                          child_idx=cix, parent_trace=self)
                        self.child_traces.append(child)
                elif trace_diff < 0:
                    for _ in range(abs(trace_diff)):
                        rm_trace = self.child_traces.pop()
                        rm_trace.remove()

                # Set parent color
                self.color = self.trace_colors[0]

            # Only for child trace
            else:
                self.color = self.parent_trace.trace_colors[
                    self.child_idx + 1]

        # Raw/ICA
        else:
            if self.isbad:
                self.setZValue(0)
                self.color = self.mne.ch_color_bad
            else:
                self.setZValue(1)
                self.color = self.mne.ch_color_ref[self.ch_name]

        self.setPen(self.mne.mkPen(_get_color(self.color, self.mne.dark)))

    @propagate_to_children
    def update_range_idx(self):  # noqa: D401
        """Should be updated when view-range or ch_idx changes."""
        self.range_idx = np.argwhere(self.mne.picks == self.ch_idx)[0][0]

    @propagate_to_children
    def update_ypos(self):  # noqa: D401
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

    @propagate_to_children
    def update_scale(self):  # noqa: D102
        transform = QTransform()
        transform.scale(1., self.mne.scale_factor)
        self.setTransform(transform)

        if self.mne.clipping is not None:
            self.update_data(propagate=False)

    @propagate_to_children
    def set_ch_idx(self, ch_idx):
        """Set the channel index and all deriving indices."""
        # The ch_idx is the index of the channel represented by this trace
        # in the channel-order from the unchanged instance (which also picks
        # refer to).
        self.ch_idx = ch_idx
        # The range_idx is the index of the channel represented by this trace
        # in the shown range.
        self.update_range_idx(propagate=False)
        # The order_idx is the index of the channel represented by this trace
        # in the channel-order (defined e.g. by group_by).
        self.order_idx = np.argwhere(self.mne.ch_order == self.ch_idx)[0][0]
        self.ch_name = self.mne.inst.ch_names[ch_idx]
        self.isbad = self.ch_name in self.mne.info['bads']
        self.ch_type = self.mne.ch_types[ch_idx]
        self.update_ypos(propagate=False)

    @propagate_to_children
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
        times = self.mne.times

        # Get decim-specific time if enabled
        if self.mne.decim != 1:
            times = times[::self.mne.decim_data[self.range_idx]]
            data = data[..., ::self.mne.decim_data[self.range_idx]]

        # For multiple color traces with epochs
        # replace values from other colors with NaN.
        if self.mne.is_epochs:
            data = np.copy(data)
            check_color = self.mne.epoch_color_ref[self.ch_idx,
                                                   self.mne.epoch_idx]
            bool_ixs = np.invert(np.equal(self.color, check_color).all(axis=1))
            starts = self.mne.boundary_times[self.mne.epoch_idx][bool_ixs]
            stops = self.mne.boundary_times[self.mne.epoch_idx + 1][bool_ixs]

            for start, stop in zip(starts, stops):
                data[np.logical_and(start <= times, times <= stop)] = np.nan

        assert times.shape[-1] == data.shape[-1]
        self.setData(times, data, connect=connect, skipFiniteCheck=skip,
                     antialias=self.mne.antialiasing)

        self.setPos(0, self.ypos)

    def toggle_bad(self, x=None):
        """Toggle bad status."""
        # Toggle bad epoch
        if self.mne.is_epochs and x is not None:
            epoch_idx, color = self.weakmain()._toggle_bad_epoch(x)

            # Update epoch color
            if color != 'none':
                new_epo_color = np.repeat(to_rgba_array(color),
                                          len(self.mne.inst.ch_names), axis=0)
            elif self.mne.epoch_colors is None:
                new_epo_color = np.concatenate(
                        [to_rgba_array(c) for c
                         in self.mne.ch_color_ref.values()])
            else:
                new_epo_color = \
                    np.concatenate([to_rgba_array(c) for c in
                                    self.mne.epoch_colors[epoch_idx]])

            # Update bad channel colors
            bad_idxs = np.in1d(self.mne.ch_names, self.mne.info['bads'])
            new_epo_color[bad_idxs] = to_rgba_array(self.mne.ch_color_bad)

            self.mne.epoch_color_ref[:, epoch_idx] = new_epo_color

            # Update overview-bar
            self.mne.overview_bar.update_bad_epochs()

            # Update other traces inlcuding self
            for trace in self.mne.traces:
                trace.update_color()
                # Update data is necessary because colored segments will vary
                trace.update_data()

        # Toggle bad channel
        else:
            bad_color, pick, marked_bad = self.weakmain()._toggle_bad_channel(
                self.range_idx)

            # Update line color status
            self.isbad = not self.isbad

            # Update colors for epochs
            if self.mne.is_epochs:
                if marked_bad:
                    new_ch_color = np.repeat(to_rgba_array(bad_color),
                                             len(self.mne.inst), axis=0)
                elif self.mne.epoch_colors is None:
                    ch_color = self.mne.ch_color_ref[self.ch_name]
                    new_ch_color = np.repeat(to_rgba_array(ch_color),
                                             len(self.mne.inst), axis=0)
                else:
                    new_ch_color = np.concatenate([to_rgba_array(c[pick]) for
                                                   c in self.mne.epoch_colors])

                self.mne.epoch_color_ref[pick, :] = new_ch_color

            # Update trace color
            self.update_color()
            if self.mne.is_epochs:
                self.update_data()

            # Update channel-axis
            self.weakmain()._update_yaxis_labels()

            # Update overview-bar
            self.mne.overview_bar.update_bad_channels()

            # Update sensor color (if in selection mode)
            if self.mne.fig_selection is not None:
                self.mne.fig_selection._update_bad_sensors(pick, marked_bad)

    def mouseClickEvent(self, ev):
        """Customize mouse click events."""
        if (not self.clickable or ev.button() != Qt.MouseButton.LeftButton
                or self.mne.annotation_mode):
            # Explicitly ignore events in annotation-mode
            ev.ignore()
            return
        if self.mouseShape().contains(ev.pos()):
            ev.accept()
            self.toggle_bad(ev.pos().x())

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
            value_idxs = np.searchsorted(self.mne.midpoints, [minVal, maxVal])
            values = self.mne.midpoints[slice(*value_idxs)]
            spacing = len(self.mne.inst.times) / self.mne.info['sfreq']
            tick_values = [(spacing, values)]
            return tick_values
        else:
            # Save _spacing for later use
            self._spacing = self.tickSpacing(minVal, maxVal, size)
            return super().tickValues(minVal, maxVal, size)

    def tickStrings(self, values, scale, spacing):
        """Customize strings of axis values."""
        if self.mne.is_epochs:
            epoch_nums = self.mne.inst.selection
            ts = epoch_nums[np.searchsorted(self.mne.midpoints, values)]
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
        self.weakmain = weakref.ref(main)
        self.mne = main.mne
        del main
        self.ch_texts = OrderedDict()
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
            tick_strings = list(
                self.weakmain()._make_butterfly_selections_dict())
        elif self.mne.butterfly:
            _, ixs, _ = np.intersect1d(DATA_CH_TYPES_ORDER,
                                       self.mne.ch_types, return_indices=True)
            ixs.sort()
            tick_strings = np.array(DATA_CH_TYPES_ORDER)[ixs]
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
                p.setPen(_get_color(
                    'black', self.mne.dark))
            elif self.mne.butterfly:
                p.setPen(_get_color(
                    self.mne.ch_color_dict[text], self.mne.dark))
            elif text in self.mne.info['bads']:
                p.setPen(_get_color(
                    self.mne.ch_color_bad, self.mne.dark))
            else:
                p.setPen(_get_color(
                    self.mne.ch_color_ref[text], self.mne.dark))
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
            y_values = np.asarray(list(self.ch_texts.values()))[:, 1, :]
            y_diff = np.abs(y_values - ypos)
            ch_idx = int(np.argmin(y_diff, axis=0)[0])
            ch_name = list(self.ch_texts)[ch_idx]
            trace = [tr for tr in self.mne.traces
                     if tr.ch_name == ch_name][0]
            if event.button() == Qt.LeftButton:
                trace.toggle_bad()
            elif event.button() == Qt.RightButton:
                self.weakmain()._create_ch_context_fig(trace.range_idx)

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
            pos = _mouse_event_position(event)
            # QPointF->QPoint for hitTestComplexControl
            pos = QPoint(int(round(pos.x())), int(round(pos.y())))
            self.initStyleOption(opt)
            control = self.style().hitTestComplexControl(
                    QStyle.CC_ScrollBar, opt, pos, self)
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
                    pos_ = pos.x()
                    sliderLength = sr.width()
                    sliderMin = gr.x()
                    sliderMax = gr.right() - sliderLength + 1
                    if (self.layoutDirection() == Qt.RightToLeft):
                        opt.upsideDown = not opt.upsideDown
                else:
                    pos_ = pos.y()
                    sliderLength = sr.height()
                    sliderMin = gr.y()
                    sliderMax = gr.bottom() - sliderLength + 1
                self.setValue(QStyle.sliderValueFromPosition(
                        self.minimum(), self.maximum(),
                        pos_ - sliderMin, sliderMax - sliderMin,
                        opt.upsideDown))
                return

        return super().mousePressEvent(event)


class TimeScrollBar(BaseScrollBar):
    """Scrolls through time."""

    def __init__(self, mne):
        super().__init__(Qt.Horizontal)
        self.mne = mne
        self.step_factor = 1
        self.setMinimum(0)
        self.setSingleStep(1)
        self.update_duration()
        self.setFocusPolicy(Qt.WheelFocus)
        # Because valueChanged is needed (captures every input to scrollbar,
        # not just sliderMoved), there has to be made a differentiation
        # between internal and external changes.
        self.external_change = False
        self.valueChanged.connect(self._time_changed)

    def _time_changed(self, value):
        if not self.external_change:
            if self.mne.is_epochs:
                # Convert Epoch index to time
                value = self.mne.boundary_times[int(value)]
            else:
                value /= self.step_factor
            self.mne.plt.setXRange(value, value + self.mne.duration,
                                   padding=0)

    def update_value(self, value):
        """Update value of the ScrollBar."""
        # Mark change as external to avoid setting
        # XRange again in _time_changed.
        self.external_change = True
        if self.mne.is_epochs:
            set_value = np.searchsorted(self.mne.midpoints, value)
        else:
            set_value = int(value * self.step_factor)
        self.setValue(set_value)
        self.external_change = False

    def update_duration(self):
        """Update bar size."""
        if self.mne.is_epochs:
            self.setPageStep(self.mne.n_epochs)
            self.setMaximum(len(self.mne.inst) - self.mne.n_epochs)
        else:
            self.setPageStep(int(self.mne.duration))
            self.step_factor = self.mne.scroll_sensitivity / self.mne.duration
            self.setMaximum(int((self.mne.xmax - self.mne.duration)
                                * self.step_factor))

    def _update_scroll_sensitivity(self):
        old_step_factor = self.step_factor
        self.update_duration()
        self.update_value(self.value() / old_step_factor)

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
        self.setSingleStep(1)
        self.update_nchan()
        self.setFocusPolicy(Qt.WheelFocus)
        # Because valueChanged is needed (captures every input to scrollbar,
        # not just sliderMoved), there has to be made a differentiation
        # between internal and external changes.
        self.external_change = False
        self.valueChanged.connect(self._channel_changed)

    def _channel_changed(self, value):
        if not self.external_change:
            if self.mne.fig_selection:
                label = list(self.mne.ch_selections)[value]
                self.mne.fig_selection._chkbx_changed(None, label)
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

    def update_nchan(self):
        """Update bar size."""
        if getattr(self.mne, 'group_by', None) in ['position', 'selection']:
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
        self._scene = QGraphicsScene()
        super().__init__(self._scene)
        assert self.scene() is self._scene
        self.weakmain = weakref.ref(main)
        self.mne = main.mne
        del main
        self.bg_img = None
        self.bg_pxmp = None
        self.bg_pxmp_item = None
        # Set minimum Size to 1/10 of display size
        min_h = int(_screen_geometry(self).height() / 10)
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

        if self.mne.is_epochs:
            # Epochs Lines
            self.epoch_line_dict = dict()
            self.update_epoch_lines()
            self.bad_epoch_rect_dict = dict()
            self.update_bad_epochs()
        else:
            # Annotations
            self.annotations_rect_dict = dict()
            self.update_annotations()

        # VLine
        self.v_line = None
        self.update_vline()

        # View Range
        self.viewrange_rect = None
        self.update_viewrange()

    def update_epoch_lines(self):
        """Update representation of epoch lines."""
        epoch_line_pen = self.mne.mkPen(color='k', width=1)
        for t in self.mne.boundary_times[1:-1]:
            top_left = self._mapFromData(t, 0)
            bottom_right = self._mapFromData(t, len(self.mne.ch_order))
            line = self.scene().addLine(QLineF(top_left, bottom_right),
                                        epoch_line_pen)
            line.setZValue(1)
            self.epoch_line_dict[t] = line

    def update_bad_channels(self):
        """Update representation of bad channels."""
        bad_set = set(self.mne.info['bads'])
        line_set = set(self.bad_line_dict)

        add_chs = bad_set.difference(line_set)
        rm_chs = line_set.difference(bad_set)

        for line_idx, ch_idx in enumerate(self.mne.ch_order):
            ch_name = self.mne.ch_names[ch_idx]
            if ch_name in add_chs:
                start = self._mapFromData(0, line_idx)
                stop = self._mapFromData(self.mne.inst.times[-1], line_idx)
                pen = _get_color(self.mne.ch_color_bad, self.mne.dark)
                line = self.scene().addLine(QLineF(start, stop), pen)
                line.setZValue(2)
                self.bad_line_dict[ch_name] = line
            elif ch_name in rm_chs:
                self.scene().removeItem(self.bad_line_dict[ch_name])
                self.bad_line_dict.pop(ch_name)

    def update_bad_epochs(self):  # noqa: D102
        bad_set = set(self.mne.bad_epochs)
        rect_set = set(self.bad_epoch_rect_dict.keys())

        add_epos = bad_set.difference(rect_set)
        rm_epos = rect_set.difference(bad_set)

        for epo_num in self.mne.inst.selection:
            if epo_num in add_epos:
                epo_idx = self.mne.inst.selection.tolist().index(epo_num)
                start, stop = self.mne.boundary_times[epo_idx:epo_idx + 2]
                top_left = self._mapFromData(start, 0)
                bottom_right = self._mapFromData(stop, len(self.mne.ch_order))
                pen = _get_color(self.mne.epoch_color_bad, self.mne.dark)
                rect = self.scene().addRect(QRectF(top_left, bottom_right),
                                            pen=pen, brush=pen)
                rect.setZValue(3)
                self.bad_epoch_rect_dict[epo_num] = rect
            elif epo_num in rm_epos:
                self.scene().removeItem(self.bad_epoch_rect_dict[epo_num])
                self.bad_epoch_rect_dict.pop(epo_num)

    def update_events(self):
        """Update representation of events."""
        if getattr(self.mne, 'event_nums', None) is not None \
                and self.mne.events_visible:
            for ev_t, ev_id in zip(self.mne.event_times, self.mne.event_nums):
                color_name = self.mne.event_color_dict[ev_id]
                color = _get_color(color_name, self.mne.dark)
                color.setAlpha(100)
                pen = self.mne.mkPen(color)
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
        rect_set = set(self.annotations_rect_dict)

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
            color = _get_color(color_name, self.mne.dark)
            color.setAlpha(150)
            pen = self.mne.mkPen(color)
            brush = mkBrush(color)
            top_left = self._mapFromData(plot_onset, 0)
            bottom_right = self._mapFromData(plot_onset + duration,
                                             len(self.mne.ch_order))
            rect = self.scene().addRect(QRectF(top_left, bottom_right),
                                        pen, brush)
            rect.setZValue(3)
            self.annotations_rect_dict[add_onset] = {'rect': rect,
                                                     'plot_onset': plot_onset,
                                                     'duration': duration,
                                                     'color': color_name}

        # Remove onsets
        for rm_onset in rm_onsets:
            self.scene().removeItem(
                self.annotations_rect_dict[rm_onset]['rect'])
            self.annotations_rect_dict.pop(rm_onset)

        # Changes
        for edit_onset in self.annotations_rect_dict:
            plot_onset = _sync_onset(self.mne.inst, edit_onset)
            annot_idx = np.where(annotations.onset == edit_onset)[0][0]
            duration = annotations.duration[annot_idx]
            rect_duration = self.annotations_rect_dict[edit_onset]['duration']
            rect = self.annotations_rect_dict[edit_onset]['rect']
            # Update changed duration
            if duration != rect_duration:
                self.annotations_rect_dict[edit_onset]['duration'] = duration
                top_left = self._mapFromData(plot_onset, 0)
                bottom_right = self._mapFromData(plot_onset + duration,
                                                 len(self.mne.ch_order))
                rect.setRect(QRectF(top_left, bottom_right))
            # Update changed color
            description = annotations.description[annot_idx]
            color_name = self.mne.annotation_segment_colors[description]
            rect_color = self.annotations_rect_dict[edit_onset]['color']
            if color_name != rect_color:
                color = _get_color(color_name, self.mne.dark)
                color.setAlpha(150)
                pen = self.mne.mkPen(color)
                brush = mkBrush(color)
                rect.setPen(pen)
                rect.setBrush(brush)

    def update_vline(self):
        """Update representation of vline."""
        if self.mne.is_epochs:
            # VLine representation not useful in epochs-mode
            pass
        # Add VLine-Representation
        elif self.mne.vline is not None:
            value = self.mne.vline.value()
            top_left = self._mapFromData(value, 0)
            bottom_right = self._mapFromData(value, len(self.mne.ch_order))
            line = QLineF(top_left, bottom_right)
            if self.v_line is None:
                pen = self.mne.mkPen('g')
                self.v_line = self.scene().addLine(line, pen)
                self.v_line.setZValue(1)
            else:
                self.v_line.setLine(line)
        # Remove VLine-Representation
        elif self.v_line is not None:
            self.scene().removeItem(self.v_line)
            self.v_line = None

    def update_viewrange(self):
        """Update representation of viewrange."""
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
            pen = self.mne.mkPen(color='g')
            brush = mkBrush(color=(0, 0, 0, 100))
            self.viewrange_rect = self.scene().addRect(rect, pen, brush)
            self.viewrange_rect.setZValue(4)
        else:
            self.viewrange_rect.setRect(rect)

    def _set_range_from_pos(self, pos):
        x, y = self._mapToData(pos)

        # Set X
        # Check boundaries
        if self.mne.is_epochs:
            if x == '-offbounds':
                epo_idx = 0
            elif x == '+offbounds':
                epo_idx = len(self.mne.inst) - self.mne.n_epochs
            else:
                epo_idx = max(x - self.mne.n_epochs // 2, 0)
            x = self.mne.boundary_times[epo_idx]
        elif x == '-offbounds':
            x = 0
        elif x == '+offbounds':
            x = self.mne.xmax - self.mne.duration
        else:
            # Move click position to middle of view range
            x -= self.mne.duration / 2
        xmin = np.clip(x, 0, self.mne.xmax - self.mne.duration)
        xmax = np.clip(xmin + self.mne.duration,
                       self.mne.duration, self.mne.xmax)

        self.mne.plt.setXRange(xmin, xmax, padding=0)

        # Set Y
        if y == '-offbounds':
            y = 0
        elif y == '+offbounds':
            y = self.mne.ymax - (self.mne.n_channels + 1)
        else:
            # Move click position to middle of view range
            y -= self.mne.n_channels / 2
        ymin = np.clip(y, 0, self.mne.ymax - (self.mne.n_channels + 1))
        ymax = np.clip(ymin + self.mne.n_channels + 1,
                       self.mne.n_channels, self.mne.ymax)
        # Check boundaries
        if self.mne.fig_selection:
            self.mne.fig_selection._scroll_to_idx(int(ymin))
        else:
            self.mne.plt.setYRange(ymin, ymax, padding=0)

    def mousePressEvent(self, event):
        """Customize mouse press events."""
        self._set_range_from_pos(event.pos())

    def mouseMoveEvent(self, event):
        """Customize mouse move events."""
        # This temporarily circumvents a bug, which only appears on windows
        # and when pyqt>=5.14.2 is installed from conda-forge.
        # It leads to receiving mouseMoveEvents all the time when the Mouse
        # is moved through the OverviewBar, even when now MouseBUtton is
        # pressed. Dragging the mouse on OverviewBar is then
        # not possible anymore.
        if not platform.system() == 'Windows':
            self._set_range_from_pos(event.pos())

    def _fit_bg_img(self):
        # Remove previous item from scene
        if (self.bg_pxmp_item is not None and
                self.bg_pxmp_item in self.scene().items()):
            self.scene().removeItem(self.bg_pxmp_item)
        # Resize Pixmap
        if self.bg_pxmp is not None:
            cnt_rect = self.contentsRect()
            self.bg_pxmp = self.bg_pxmp.scaled(cnt_rect.width(),
                                               cnt_rect.height(),
                                               Qt.IgnoreAspectRatio)
            self.bg_pxmp_item = self.scene().addPixmap(self.bg_pxmp)

    def resizeEvent(self, event):
        """Customize resize event."""
        super().resizeEvent(event)
        cnt_rect = self.contentsRect()
        self.setSceneRect(QRectF(QPointF(0, 0),
                                 QPointF(cnt_rect.width(),
                                         cnt_rect.height())))
        # Resize backgounrd
        self._fit_bg_img()

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

        if self.mne.is_epochs:
            # Resize epoch lines
            for epo_t, epoch_line in self.epoch_line_dict.items():
                top_left = self._mapFromData(epo_t, 0)
                bottom_right = self._mapFromData(epo_t,
                                                 len(self.mne.ch_order))
                epoch_line.setLine(QLineF(top_left, bottom_right))
            # Resize bad rects
            for epo_idx, epoch_rect in self.bad_epoch_rect_dict.items():
                start, stop = self.mne.boundary_times[epo_idx:epo_idx + 2]
                top_left = self._mapFromData(start, 0)
                bottom_right = self._mapFromData(stop, len(self.mne.ch_order))
                epoch_rect.setRect(QRectF(top_left, bottom_right))
        else:
            # Resize annotation-rects
            for annot_dict in self.annotations_rect_dict.values():
                annot_rect = annot_dict['rect']
                plot_onset = annot_dict['plot_onset']
                duration = annot_dict['duration']

                top_left = self._mapFromData(plot_onset, 0)
                bottom_right = self._mapFromData(plot_onset + duration,
                                                 len(self.mne.ch_order))
                annot_rect.setRect(QRectF(top_left, bottom_right))

        # Update vline
        if all([i is not None for i in [self.v_line, self.mne.vline]]):
            value = self.mne.vline.value()
            top_left = self._mapFromData(value, 0)
            bottom_right = self._mapFromData(value, len(self.mne.ch_order))
            self.v_line.setLine(QLineF(top_left, bottom_right))

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
        self.bg_pxmp = None
        if self.mne.overview_mode == 'empty':
            pass
        elif self.mne.overview_mode == 'channels':
            channel_rgba = np.empty((len(self.mne.ch_order),
                                     2, 4))
            for line_idx, ch_idx in enumerate(self.mne.ch_order):
                ch_type = self.mne.ch_types[ch_idx]
                color = _get_color(
                    self.mne.ch_color_dict[ch_type], self.mne.dark)
                channel_rgba[line_idx, :] = color.getRgb()

            channel_rgba = np.require(channel_rgba, np.uint8, 'C')
            self.bg_img = QImage(channel_rgba,
                                 channel_rgba.shape[1],
                                 channel_rgba.shape[0],
                                 QImage.Format_RGBA8888)
            self.bg_pxmp = QPixmap.fromImage(self.bg_img)

        elif self.mne.overview_mode == 'zscore' and \
                self.mne.zscore_rgba is not None:
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
            if self.mne.is_epochs:
                # Return epoch index for epochs
                x = int(len(self.mne.inst) * xnorm)
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

    def keyPressEvent(self, event):  # noqa: D102
        self.weakmain().keyPressEvent(event)


class RawViewBox(ViewBox):
    """PyQtGraph-Wrapper for interaction with the View."""

    def __init__(self, main):
        super().__init__(invertY=True)
        self.enableAutoRange(enable=False, x=False, y=False)
        self.weakmain = weakref.ref(main)
        self.mne = main.mne
        del main
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
                    self._drag_start = 0 if self._drag_start < 0 else self._drag_start
                    drag_stop = self.mapSceneToView(event.scenePos()).x()
                    self._drag_region = AnnotRegion(self.mne,
                                                    description=description,
                                                    values=(self._drag_start,
                                                            drag_stop))
                elif event.isFinish():
                    drag_stop = self.mapSceneToView(event.scenePos()).x()
                    drag_stop = 0 if drag_stop < 0 else drag_stop
                    drag_stop = (
                        self.mne.xmax if self.mne.xmax < drag_stop else drag_stop
                    )
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
                        self.weakmain()._remove_region(
                            rm_region, from_annot=False)
                    self.weakmain()._add_region(
                        plot_onset, duration, self.mne.current_description,
                        region=self._drag_region)
                    self._drag_region.select(True)
                    self._drag_region.setZValue(2)

                    # Update Overview-Bar
                    self.mne.overview_bar.update_annotations()
                else:
                    x_to = self.mapSceneToView(event.scenePos()).x()
                    self._drag_region.setRegion((self._drag_start, x_to))

            elif event.isFinish():
                self.weakmain().message_box(
                    text='No description!',
                    info_text='No description is given, add one!',
                    icon=QMessageBox.Warning)

    def mouseClickEvent(self, event):
        """Customize mouse click events."""
        # If we want the context-menu back, uncomment following line
        # super().mouseClickEvent(event)
        if not self.mne.annotation_mode:
            if event.button() == Qt.LeftButton:
                self.weakmain()._add_vline(self.mapSceneToView(
                        event.scenePos()).x())
            elif event.button() == Qt.RightButton:
                self.weakmain()._remove_vline()

    def wheelEvent(self, ev, axis=None):
        """Customize mouse wheel/trackpad-scroll events."""
        ev.accept()
        scroll = -1 * ev.delta() / 120
        if ev.orientation() == Qt.Horizontal:
            self.weakmain().hscroll(scroll * 10)
        elif ev.orientation() == Qt.Vertical:
            self.weakmain().vscroll(scroll)

    def keyPressEvent(self, event):  # noqa: D102
        self.weakmain().keyPressEvent(event)


class VLineLabel(InfLineLabel):
    """Label of the vline displaying the time."""

    def __init__(self, vline):
        super().__init__(vline, text='{value:.3f} s', position=0.98,
                         fill='g', color='b', movable=True)
        self.cursorOffset = None

    def mouseDragEvent(self, ev):
        """Customize mouse drag events."""
        if self.movable and ev.button() == Qt.LeftButton:
            if ev.isStart():
                self.line.moving = True
                self.cursorOffset = (self.line.pos() -
                                     self.mapToView(ev.buttonDownPos()))
            ev.accept()

            if not self.line.moving:
                return

            self.line.setPos(self.cursorOffset + self.mapToView(ev.pos()))
            self.line.sigDragged.emit(self)
            if ev.isFinish():
                self.line.moving = False
                self.line.sigPositionChangeFinished.emit(self.line)

    def valueChanged(self):
        """Customize what happens on value change."""
        if not self.isVisible():
            return
        value = self.line.value()
        if self.line.mne.is_epochs:
            # Show epoch-time
            t_vals_abs = np.linspace(0, self.line.mne.epoch_dur,
                                     len(self.line.mne.inst.times))
            search_val = value % self.line.mne.epoch_dur
            t_idx = np.searchsorted(t_vals_abs, search_val)
            value = self.line.mne.inst.times[t_idx]
        self.setText(self.format.format(value=value))
        self.updatePosition()


class VLine(InfiniteLine):
    """Marker to be placed inside the Trace-Plot."""

    def __init__(self, mne, pos, bounds):
        super().__init__(pos, pen='g', hoverPen='y',
                         movable=True, bounds=bounds)
        self.mne = mne
        self.label = VLineLabel(self)


def _q_font(point_size, bold=False):
    font = QFont()
    font.setPointSize(point_size)
    font.setBold(bold)
    return font


class EventLine(InfiniteLine):
    """Displays Events inside Trace-Plot."""

    def __init__(self, mne, pos, label, color):
        super().__init__(pos, pen=color, movable=False,
                         label=str(label), labelOpts={'position': 0.98,
                                                      'color': color,
                                                      'anchors': [(0, 0.5),
                                                                  (0, 0.5)]})
        self.mne = mne
        self.label.setFont(_q_font(10, bold=True))
        self.setZValue(0)

        self.mne.plt.addItem(self)


class Crosshair(InfiniteLine):
    """Continously updating marker inside the Trace-Plot."""

    def __init__(self, mne):
        super().__init__(angle=90, movable=False, pen='g')
        self.mne = mne
        self.y = 1

    def set_data(self, x, y):
        """Set x and y data for crosshair point."""
        self.setPos(x)
        self.y = y

    def paint(self, p, *args):  # noqa: D102
        super().paint(p, *args)

        p.setPen(self.mne.mkPen('r', width=4))
        p.drawPoint(Point(self.y, 0))


class BaseScaleBar:  # noqa: D101
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
            # Consider all indices bad
            if self.ypos is None:
                self.ypos = self.mne.ch_start + ch_type_idxs[0] + 1

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


class ScaleBarText(BaseScaleBar, TextItem):  # noqa: D101
    def __init__(self, mne, ch_type):
        BaseScaleBar.__init__(self, mne, ch_type)
        TextItem.__init__(self, color='#AA3377')

        self.setFont(_q_font(10))
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


class ScaleBar(BaseScaleBar, QGraphicsLineItem):  # noqa: D101
    def __init__(self, mne, ch_type):
        BaseScaleBar.__init__(self, mne, ch_type)
        QGraphicsLineItem.__init__(self)

        self.setZValue(1)
        self.setPen(self.mne.mkPen(color='#AA3377', width=5))
        self.update_y_position()

    def _set_position(self, x, y):
        self.setLine(QLineF(x, y - 0.5, x, y + 0.5))

    def get_ydata(self):
        """Get y-data for tests."""
        line = self.line()
        return line.y1(), line.y2()


class _BaseDialog(QDialog):
    def __init__(self, main, widget=None,
                 modal=False, name=None, title=None,
                 flags=Qt.Window | Qt.Tool):
        super().__init__(main, flags)
        self.weakmain = weakref.ref(main)
        self.widget = widget
        self.mne = main.mne
        del main
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
            cp = _screen_geometry(self).center()
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

    # If this widget gets activated (e.g., the user clicks away from the
    # browser but then returns to it by clicking in a selection window),
    # the main window should be raised as well
    def event(self, event):
        if event.type() == QEvent.WindowActivate:
            self.weakmain().raise_()
        return super().event(event)


class SettingsDialog(_BaseDialog):
    """Shows additional settings."""

    def __init__(self, main, title='Settings', **kwargs):
        super().__init__(main, title=title, **kwargs)

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
        self.downsampling_box.valueChanged.connect(_methpartial(
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
        self.ds_method_cmbx.currentTextChanged.connect(
            _methpartial(self._value_changed, value_name='ds_method'))
        self.ds_method_cmbx.setCurrentText(self.mne.ds_method)
        layout.addRow('ds_method', self.ds_method_cmbx)

        self.scroll_sensitivity_slider = QSlider(Qt.Horizontal)
        self.scroll_sensitivity_slider.setMinimum(10)
        self.scroll_sensitivity_slider.setMaximum(1000)
        self.scroll_sensitivity_slider.setToolTip('Set the sensitivity of '
                                                  'the scrolling in '
                                                  'horizontal direction.')
        self.scroll_sensitivity_slider.valueChanged.connect(
            _methpartial(self._value_changed, value_name='scroll_sensitivity'))
        # Set default
        self.scroll_sensitivity_slider.setValue(self.mne.scroll_sensitivity)
        layout.addRow('horizontal scroll sensitivity',
                      self.scroll_sensitivity_slider)
        self.setLayout(layout)
        self.show()

    def closeEvent(self, event):  # noqa: D102
        _disconnect(self.ds_method_cmbx.currentTextChanged)
        _disconnect(self.scroll_sensitivity_slider.valueChanged)
        super().closeEvent(event)

    def _value_changed(self, new_value, value_name):
        if value_name == 'downsampling' and new_value == 0:
            new_value = 'auto'

        setattr(self.mne, value_name, new_value)

        if value_name == 'scroll_sensitivity':
            self.mne.ax_hscroll._update_scroll_sensitivity()
        else:
            self.weakmain()._redraw()


class HelpDialog(_BaseDialog):
    """Shows all keyboard-shortcuts."""

    def __init__(self, main, **kwargs):
        super().__init__(main, title='Help', **kwargs)

        # Show all keyboard-shortcuts in a Scroll-Area
        layout = QVBoxLayout()
        keyboard_label = QLabel('Keyboard Shortcuts')
        keyboard_label.setFont(_q_font(16, bold=True))
        layout.addWidget(keyboard_label)

        scroll_area = QScrollArea()
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
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

        # Additional help for mouse interaction
        inst = self.mne.instance_type
        is_raw = inst == 'raw'
        is_epo = inst == 'epochs'
        is_ica = inst == 'ica'
        ch_cmp = 'component' if is_ica else 'channel'
        ch_epo = 'epoch' if is_epo else 'channel'
        ica_bad = 'Mark/unmark component for exclusion'
        lclick_data = ica_bad if is_ica else f'Mark/unmark bad {ch_epo}'
        lclick_name = (ica_bad if is_ica else 'Mark/unmark bad channel')
        ldrag = 'add annotation (in annotation mode)' if is_raw else None
        rclick_name = dict(ica='Show diagnostics for component',
                           epochs='Show imageplot for channel',
                           raw='Show channel location')[inst]
        mouse_help = [(f'Left-click {ch_cmp} name', lclick_name),
                      (f'Left-click {ch_cmp} data', lclick_data),
                      ('Left-click-and-drag on plot', ldrag),
                      ('Left-click on plot background',
                       'Place vertical guide'),
                      ('Right-click on plot background',
                       'Clear vertical guide'),
                      ('Right-click on channel name', rclick_name)]

        mouse_label = QLabel('Mouse Interaction')
        mouse_label.setFont(_q_font(16, bold=True))
        layout.addWidget(mouse_label)
        mouse_widget = QWidget()
        mouse_layout = QFormLayout()
        for interaction, description in mouse_help:
            if description is not None:
                mouse_layout.addRow(f'{interaction}:', QLabel(description))
        mouse_widget.setLayout(mouse_layout)
        layout.addWidget(mouse_widget)

        self.setLayout(layout)
        self.show()

        # Set minimum width to avoid horizontal scrolling
        scroll_area.setMinimumWidth(scroll_widget.minimumSizeHint().width() +
                                    scroll_area.verticalScrollBar().width())
        self.update()


class ProjDialog(_BaseDialog):
    """A dialog to toggle projections."""

    def __init__(self, main, *, name):
        self.external_change = True
        # Create projection-layout
        super().__init__(main.window(), name=name, title='Projectors')

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
            chkbx.clicked.connect(_methpartial(self._proj_changed, idx=idx))
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
            self.weakmain()._apply_update_projectors()

    def toggle_all(self):
        """Toggle all projectors."""
        self.weakmain()._apply_update_projectors(toggle_all=True)

        # Update all checkboxes
        for idx, chkbx in enumerate(self.checkboxes):
            chkbx.setChecked(bool(self.mne.projs_on[idx]))


class _ChannelFig(FigureCanvasQTAgg):
    def __init__(self, figure, mne):
        self.figure = figure
        self.mne = mne
        super().__init__(figure)
        self.setFocusPolicy(Qt.FocusPolicy(Qt.StrongFocus | Qt.WheelFocus))
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
            painter.setPen(self.mne.mkPen('red', width=2))
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


class SelectionDialog(_BaseDialog):  # noqa: D101
    def __init__(self, main):
        # Create widget
        super().__init__(main, name='fig_selection',
                         title='Channel selection')
        geo = _screen_geometry(self)
        # Position selection dialog at right border of active screen
        xpos = geo.x() + geo.width() - 400
        self.setGeometry(xpos, 100, 400, 800)

        layout = QVBoxLayout()

        # Add channel plot
        fig = _figure_agg(figsize=(6, 6), dpi=96)
        ax = fig.add_axes([0, 0, 1, 1])
        self.channel_fig = plot_sensors(self.mne.info, kind='select',
                                        ch_type='all', title='',
                                        ch_groups=self.mne.group_by, axes=ax,
                                        show=False)[0]
        self.channel_fig.lasso.callbacks.append(self._set_custom_selection)
        self.channel_widget = _ChannelFig(self.channel_fig, self.mne)
        layout.addWidget(self.channel_widget)

        selections_dict = self.mne.ch_selections
        selections_dict.update(Custom=np.array([], dtype=int))  # for lasso

        self.chkbxs = OrderedDict()
        for label in selections_dict:
            chkbx = QRadioButton(label)
            chkbx.clicked.connect(
                _methpartial(self._chkbx_changed, label=label))
            self.chkbxs[label] = chkbx
            layout.addWidget(chkbx)

        self.mne.old_selection = list(selections_dict)[0]
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

    def _chkbx_changed(self, checked=True, label=None):
        # _chkbx_changed is called either directly (with checked=None) or
        # through _methpartial with a Qt signal. The signal includes the bool
        # argument 'checked'.
        # Old versions of MNE-python tests will call this function directly
        # without the checked argument _chkbx_changed(label), thus it has to be
        # wrap in case only one argument is provided to retain compatibility
        # of the tests between new/old versions of mne-qt-browser and
        # mne-python.
        if label is None:
            label = checked
        # Disable butterfly if checkbox is clicked
        if self.mne.butterfly:
            self.weakmain()._set_butterfly(False)
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
        label_idx = list(self.mne.ch_selections).index(label)
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
            self._chkbx_changed(None, 'Custom')

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
        name_idx = list(self.mne.ch_selections).index(
                self.mne.old_selection)
        new_idx = np.clip(name_idx + step,
                          0, len(self.mne.ch_selections) - 1)
        new_label = list(self.mne.ch_selections)[new_idx]
        self._chkbx_changed(None, new_label)

    def _scroll_to_idx(self, idx):
        all_values = list()
        label = list(self.mne.ch_selections)[0]
        for key, values in self.mne.ch_selections.items():
            all_values = np.concatenate([all_values, values])
            if idx < len(all_values):
                label = key
                break
        self._chkbx_changed(None, label)

    def closeEvent(self, event):  # noqa: D102
        super().closeEvent(event)
        if hasattr(self.channel_fig.lasso, 'callbacks'):
            # MNE >= 1.0
            self.channel_fig.lasso.callbacks.clear()
        for chkbx in self.chkbxs.values():
            _disconnect(chkbx.clicked, allow_error=True)
        main = self.weakmain()
        if main is not None:
            main.close()


class AnnotRegion(LinearRegionItem):
    """Graphics-Oobject for Annotations."""

    regionChangeFinished = Signal(object)
    gotSelected = Signal(object)
    removeRequested = Signal(object)

    def __init__(self, mne, description, values):
        super().__init__(values=values, orientation='vertical',
                         movable=True, swapMode='sort',
                         bounds=(0, mne.xmax))
        # Set default z-value to 0 to be behind other items in scene
        self.setZValue(0)

        self.sigRegionChangeFinished.connect(self._region_changed)
        self.mne = mne
        self.description = description
        self.old_onset = values[0]
        self.selected = False

        self.label_item = TextItem(text=description, anchor=(0.5, 0.5))
        self.label_item.setFont(_q_font(10, bold=True))
        self.sigRegionChanged.connect(self.update_label_pos)

        self.update_color()

        self.mne.plt.addItem(self, ignoreBounds=True)
        self.mne.plt.addItem(self.label_item, ignoreBounds=True)

    def _region_changed(self):
        self.regionChangeFinished.emit(self)
        self.old_onset = self.getRegion()[0]

    def update_color(self):
        """Update color of annotation-region."""
        color_string = self.mne.annotation_segment_colors[self.description]
        self.base_color = _get_color(color_string, self.mne.dark)
        self.hover_color = _get_color(color_string, self.mne.dark)
        self.text_color = _get_color(color_string, self.mne.dark)
        self.base_color.setAlpha(75)
        self.hover_color.setAlpha(150)
        self.text_color.setAlpha(255)
        self.line_pen = self.mne.mkPen(color=self.hover_color, width=2)
        self.hover_pen = self.mne.mkPen(color=self.text_color, width=2)
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
                # the annotation removed should be the one on top of all others, which
                # should correspond to the one of the type currently selected and with
                # the highest zValue
                event.accept()
        else:
            event.ignore()

    def mouseDragEvent(self, ev):
        """Customize mouse drag events."""
        if (
            not self.mne.annotation_mode
            or not self.movable
            or not ev.button() == Qt.LeftButton
        ):
            return
        ev.accept()

        if ev.isStart():
            bdp = ev.buttonDownPos()
            self.cursorOffsets = [line.pos() - bdp for line in self.lines]
            self.startPositions = [line.pos() for line in self.lines]
            self.moving = True

        if not self.moving:
            return

        new_pos = [pos + ev.pos() for pos in self.cursorOffsets]
        # make sure the new_pos is not exiting the boundaries set for each line which
        # corresponds to (0, raw.times[-1])
        # we have to take into account regions draw from right to left and from left to
        # right separately because we are changing the position of the individual lines
        # used to create the region
        idx = 0 if new_pos[0].x() <= new_pos[1].x() else 1
        if new_pos[idx].x() < self.lines[idx].bounds()[0]:
            shift = self.lines[idx].bounds()[0] - new_pos[idx].x()
            for pos in new_pos:
                pos.setX(pos.x() + shift)
        if self.lines[(idx + 1) % 2].bounds()[1] < new_pos[(idx + 1) % 2].x():
            shift = new_pos[(idx + 1) % 2].x() - self.lines[(idx + 1) % 2].bounds()[1]
            for pos in new_pos:
                pos.setX(pos.x() - shift)

        with SignalBlocker(self.lines[0]):
            for pos, line in zip(new_pos, self.lines):
                line.setPos(pos)
        self.prepareGeometryChange()

        if ev.isFinish():
            self.moving = False
            self.sigRegionChangeFinished.emit(self)
        else:
            self.sigRegionChanged.emit(self)

    def update_label_pos(self):
        """Update position of description-label from annotation-region."""
        rgn = self.getRegion()
        vb = self.mne.viewbox
        if vb:
            ymax = vb.viewRange()[1][1]
            self.label_item.setPos(sum(rgn) / 2, ymax - 0.3)


class _AnnotEditDialog(_BaseDialog):
    def __init__(self, annot_dock):
        super().__init__(annot_dock.weakmain(), title='Edit Annotations')
        self.ad = annot_dock

        self.current_mode = None

        layout = QVBoxLayout()
        self.descr_label = QLabel()
        if self.mne.selected_region:
            self.mode_cmbx = QComboBox()
            self.mode_cmbx.addItems(['all', 'selected'])
            self.mode_cmbx.currentTextChanged.connect(self._mode_changed)
            layout.addWidget(QLabel('Edit Scope:'))
            layout.addWidget(self.mode_cmbx)
        # Set group as default
        self._mode_changed('all')

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
        if mode == 'all':
            curr_des = self.ad.description_cmbx.currentText()
        else:
            curr_des = self.mne.selected_region.description
        self.descr_label.setText(f'Change "{curr_des}" to:')

    def _edit(self):
        new_des = self.input_w.text()
        if new_des:
            if self.current_mode == 'all' or self.mne.selected_region is None:
                self.ad._edit_description_all(new_des)
            else:
                self.ad._edit_description_selected(new_des)
            self.close()


def _select_all(chkbxs):
    for chkbx in chkbxs:
        chkbx.setChecked(True)


def _clear_all(chkbxs):
    for chkbx in chkbxs:
        chkbx.setChecked(False)


class AnnotationDock(QDockWidget):
    """Dock-Window for Management of annotations."""

    def __init__(self, main):
        super().__init__('Annotations')
        self.weakmain = weakref.ref(main)
        self.mne = main.mne
        del main
        self._init_ui()

        self.setFeatures(QDockWidget.DockWidgetMovable |
                         QDockWidget.DockWidgetFloatable)

    def _init_ui(self):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignLeft)

        self.description_cmbx = QComboBox()
        self.description_cmbx.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.description_cmbx.currentIndexChanged.connect(self._description_changed)
        self._update_description_cmbx()
        layout.addWidget(self.description_cmbx)

        add_bt = QPushButton('Add Description')
        add_bt.clicked.connect(self._add_description_dlg)
        layout.addWidget(add_bt)

        rm_bt = QPushButton('Remove Description')
        rm_bt.clicked.connect(self._remove_description_dlg)
        layout.addWidget(rm_bt)

        edit_bt = QPushButton('Edit Description')
        edit_bt.clicked.connect(self._edit_description_dlg)
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
        self.start_bx.setMinimum(0)
        self.start_bx.setMaximum(self.mne.xmax - 1 / self.mne.info["sfreq"])
        self.start_bx.setSingleStep(0.05)
        self.start_bx.valueChanged.connect(self._start_changed)
        layout.addWidget(self.start_bx)

        layout.addWidget(QLabel('Stop:'))
        self.stop_bx = QDoubleSpinBox()
        self.stop_bx.setDecimals(time_decimals)
        self.stop_bx.setMinimum(1 / self.mne.info["sfreq"])
        self.stop_bx.setMaximum(self.mne.xmax)
        self.stop_bx.setSingleStep(0.05)
        self.stop_bx.valueChanged.connect(self._stop_changed)
        layout.addWidget(self.stop_bx)

        help_bt = QPushButton(QIcon.fromTheme("help"), 'Help')
        help_bt.clicked.connect(self._show_help)
        layout.addWidget(help_bt)

        widget.setLayout(layout)
        self.setWidget(widget)

    def _add_description_to_cmbx(self, description):
        color_pixmap = QPixmap(25, 25)
        color = _get_color(
            self.mne.annotation_segment_colors[description], self.mne.dark)
        color.setAlpha(75)
        color_pixmap.fill(color)
        color_icon = QIcon(color_pixmap)
        self.description_cmbx.addItem(color_icon, description)

    def _add_description(self, new_description):
        self.mne.new_annotation_labels.append(new_description)
        self.mne.visible_annotations[new_description] = True
        self.weakmain()._setup_annotation_colors()
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

    def _edit_description_all(self, new_des):
        """Update descriptions of all annotations with the same description."""
        old_des = self.description_cmbx.currentText()
        edit_regions = [r for r in self.mne.regions
                        if r.description == old_des]
        # Update regions & annotations
        for ed_region in edit_regions:
            idx = self.weakmain()._get_onset_idx(ed_region.getRegion()[0])
            self.mne.inst.annotations.description[idx] = new_des
            ed_region.update_description(new_des)
        # Update containers with annotation-attributes
        self.mne.new_annotation_labels.remove(old_des)
        self.mne.new_annotation_labels = \
            self.weakmain()._get_annotation_labels()
        self.mne.visible_annotations[new_des] = \
            self.mne.visible_annotations.pop(old_des)
        self.mne.annotation_segment_colors[new_des] = \
            self.mne.annotation_segment_colors.pop(old_des)

        # Update related widgets
        self.weakmain()._setup_annotation_colors()
        self._update_regions_colors()
        self._update_description_cmbx()
        self.mne.current_description = new_des
        self.mne.overview_bar.update_annotations()

    def _edit_description_selected(self, new_des):
        """Update description only of selected region."""
        old_des = self.mne.selected_region.description
        idx = self.weakmain()._get_onset_idx(
            self.mne.selected_region.getRegion()[0])
        # Update regions & annotations
        self.mne.inst.annotations.description[idx] = new_des
        self.mne.selected_region.update_description(new_des)
        # Update containers with annotation-attributes
        if new_des not in self.mne.new_annotation_labels:
            self.mne.new_annotation_labels.append(new_des)
        self.mne.visible_annotations[new_des] = \
            copy(self.mne.visible_annotations[old_des])
        if old_des not in self.mne.inst.annotations.description:
            self.mne.new_annotation_labels.remove(old_des)
            self.mne.visible_annotations.pop(old_des)
            self.mne.annotation_segment_colors[new_des] = \
                self.mne.annotation_segment_colors.pop(old_des)

        # Update related widgets
        self.weakmain()._setup_annotation_colors()
        self._update_regions_colors()
        self._update_description_cmbx()
        self.mne.overview_bar.update_annotations()

    def _edit_description_dlg(self):
        if len(self.mne.inst.annotations.description) > 0:
            _AnnotEditDialog(self)
        else:
            self.weakmain().message_box(
                text='No Annotations!',
                info_text='There are no annotations yet to edit!',
                icon=QMessageBox.Information)

    def _remove_description(self, rm_description):
        # Remove regions
        for rm_region in [r for r in self.mne.regions
                          if r.description == rm_description]:
            rm_region.remove()

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

    def _remove_description_dlg(self):
        rm_description = self.description_cmbx.currentText()
        existing_annot = list(self.mne.inst.annotations.description).count(
                rm_description)
        if existing_annot > 0:
            text = f'Remove annotations with {rm_description}?'
            info_text = f'There exist {existing_annot} annotations with ' \
                        f'"{rm_description}".\n' \
                        f'Do you really want to remove them?'
            buttons = QMessageBox.Yes | QMessageBox.No
            ans = self.weakmain().message_box(
                text=text, info_text=info_text, buttons=buttons,
                default_button=QMessageBox.Yes, icon=QMessageBox.Question)
        else:
            ans = QMessageBox.Yes

        if ans == QMessageBox.Yes:
            self._remove_description(rm_description)

    def _set_visible_region(self, state, *, description):
        self.mne.visible_annotations[description] = bool(state)

    def _select_annotations(self):
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
            chkbx.stateChanged.connect(
                _methpartial(self._set_visible_region, description=des))
            chkbxs.append(chkbx)
            scroll_layout.addWidget(chkbx)

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        bt_layout = QGridLayout()

        all_bt = QPushButton('All')
        all_bt.clicked.connect(partial(_select_all, chkbxs=chkbxs))
        bt_layout.addWidget(all_bt, 0, 0)

        clear_bt = QPushButton('Clear')
        clear_bt.clicked.connect(partial(_clear_all, chkbxs=chkbxs))
        bt_layout.addWidget(clear_bt, 0, 1)

        ok_bt = QPushButton('Ok')
        ok_bt.clicked.connect(select_dlg.close)
        bt_layout.addWidget(ok_bt, 1, 0, 1, 2)

        layout.addLayout(bt_layout)

        select_dlg.setLayout(layout)
        select_dlg.exec()
        all_bt.clicked.disconnect()
        clear_bt.clicked.disconnect()

        self.weakmain()._update_regions_visible()

    def _description_changed(self, descr_idx):
        new_descr = self.description_cmbx.itemText(descr_idx)
        self.mne.current_description = new_descr
        # increase zValue of currently selected annotation and decrease all the others
        for region in self.mne.regions:
            if region.description == self.mne.current_description:
                region.setZValue(2)
            else:
                region.setZValue(1)

    def _start_changed(self):
        start = self.start_bx.value()
        sel_region = self.mne.selected_region
        stop = sel_region.getRegion()[1]
        if start < stop:
            self.mne.selected_region.setRegion((start, stop))
        else:
            self.weakmain().message_box(
                text='Invalid value!',
                info_text='Start can\'t be bigger or equal to Stop!',
                icon=QMessageBox.Critical, modal=False)
            self.start_bx.setValue(sel_region.getRegion()[0])

    def _stop_changed(self):
        stop = self.stop_bx.value()
        sel_region = self.mne.selected_region
        start = sel_region.getRegion()[0]
        if start < stop:
            sel_region.setRegion((start, stop))
        else:
            self.weakmain().message_box(
                text='Invalid value!',
                info_text='Stop can\'t be smaller or equal to Start!',
                icon=QMessageBox.Critical)
            self.stop_bx.setValue(sel_region.getRegion()[1])

    def _set_color(self):
        curr_descr = self.description_cmbx.currentText()
        if curr_descr in self.mne.annotation_segment_colors:
            curr_col = self.mne.annotation_segment_colors[curr_descr]
        else:
            curr_col = None
        color = QColorDialog.getColor(
            _get_color(curr_col, self.mne.dark), self,
            f'Choose color for {curr_descr}!')
        if color.isValid():
            # Invert it (we only want to display inverted colors, all stored
            # colors should be for light mode)
            color = _get_color(color.getRgb(), self.mne.dark)
            self.mne.annotation_segment_colors[curr_descr] = color
            self._update_regions_colors()
            self._update_description_cmbx()
            self.mne.overview_bar.update_annotations()

    def update_values(self, region):
        """Update spinbox-values from region."""
        rgn = region.getRegion()
        self.start_bx.setEnabled(True)
        self.stop_bx.setEnabled(True)
        with SignalBlocker(self.start_bx):
            self.start_bx.setValue(rgn[0])
        with SignalBlocker(self.stop_bx):
            self.stop_bx.setValue(rgn[1])

    def _update_description_cmbx(self):
        self.description_cmbx.clear()
        descriptions = self.weakmain()._get_annotation_labels()
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
        with SignalBlocker(self.start_bx):
            self.start_bx.setValue(0)
        with SignalBlocker(self.stop_bx):
            self.stop_bx.setValue(1 / self.mne.info["sfreq"])

    def _show_help(self):
        info_text = '<h1>Help</h1>' \
                    '<h2>Annotations</h2>' \
                    '<h3>Add Annotations</h3>' \
                    'Drag inside the data-view to create annotations with '\
                    'the description currently selected (leftmost item of '\
                    'the toolbar).If there is no description yet, add one ' \
                    'with the button "Add description".' \
                    '<h3>Remove Annotations</h3>' \
                    'You can remove single annotations by right-clicking on '\
                    'them.' \
                    '<h3>Edit Annotations</h3>' \
                    'You can edit annotations by dragging them or their '\
                    'boundaries. Or you can use the dials in the toolbar to '\
                    'adjust the boundaries for the current selected '\
                    'annotation.' \
                    '<h2>Descriptions</h2>' \
                    '<h3>Add Description</h3>' \
                    'Add a new description with ' \
                    'the button "Add description".' \
                    '<h3>Edit Description</h3>' \
                    'You can edit the description of one single annotation '\
                    'or all annotations of the currently selected kind with '\
                    'the button "Edit description".' \
                    '<h3>Remove Description</h3>' \
                    'You can remove all annotations of the currently '\
                    'selected kind with the button "Remove description".'
        self.weakmain().message_box(
            text='Annotations-Help', info_text=info_text,
            icon=QMessageBox.Information)


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
        self.sigSceneMouseMoved.emit(_mouse_event_position(ev))


def _mouse_event_position(ev):
    try:  # Qt6
        return ev.position()
    except AttributeError:
        return ev.pos()


class LoadThread(QThread):
    """A worker object for precomputing in a separate QThread."""

    loadProgress = Signal(int)
    processText = Signal(str)
    loadingFinished = Signal()

    def __init__(self, browser):
        super().__init__()
        self.weakbrowser = weakref.ref(browser)
        self.mne = browser.mne
        self.loadProgress.connect(self.mne.load_progressbar.setValue)
        self.processText.connect(
            _methpartial(browser._show_process))
        self.loadingFinished.connect(
            _methpartial(browser._precompute_finished))

    def run(self):
        """Load and process data in a separate QThread."""
        # Split data loading into 10 chunks to show user progress.
        # Testing showed that e.g. n_chunks=100 extends loading time
        # (at least for the sample dataset)
        # because of the frequent gui-update-calls.
        # Thus n_chunks = 10 should suffice.
        data = None
        if self.mne.is_epochs:
            times = np.arange(len(self.mne.inst) * len(self.mne.inst.times)) \
                    / self.mne.info['sfreq']
        else:
            times = None
        n_chunks = min(10, len(self.mne.inst))
        chunk_size = len(self.mne.inst) // n_chunks
        browser = self.weakbrowser()
        for n in range(n_chunks):
            start = n * chunk_size
            if n == n_chunks - 1:
                # Get last chunk which may be larger due to rounding above
                stop = None
            else:
                stop = start + chunk_size
            # Load epochs
            if self.mne.is_epochs:
                item = slice(start, stop)
                with self.mne.inst.info._unlock():
                    data_chunk = np.concatenate(
                            self.mne.inst.get_data(item=item), axis=-1)
            # Load raw
            else:
                data_chunk, times_chunk = browser._load_data(start, stop)
                if times is None:
                    times = times_chunk
                else:
                    times = np.concatenate((times, times_chunk), axis=0)

            if data is None:
                data = data_chunk
            else:
                data = np.concatenate((data, data_chunk), axis=1)

            self.loadProgress.emit(n + 1)

        picks = self.mne.ch_order
        # Deactive remove dc because it will be removed for visible range
        stashed_remove_dc = self.mne.remove_dc
        self.mne.remove_dc = False
        data = browser._process_data(data, 0, data.shape[-1], picks, self)
        self.mne.remove_dc = stashed_remove_dc

        self.mne.global_data = data
        self.mne.global_times = times

        # Calculate Z-Scores
        self.processText.emit('Calculating Z-Scores...')
        browser._get_zscore(data)
        del browser

        self.loadingFinished.emit()

    def clean(self):  # noqa: D102
        if self.isRunning():
            wait_time = 10  # max. waiting time in seconds
            logger.info('Waiting for Loading-Thread to finish... '
                        f'(max. {wait_time} sec)')
            self.wait(int(wait_time * 1e3))
        _disconnect(self.loadProgress)
        _disconnect(self.processText)
        _disconnect(self.loadingFinished)
        del self.mne
        del self.weakbrowser


class _PGMetaClass(type(QMainWindow), type(BrowserBase)):
    """Class is necessary to prevent a metaclass conflict.

    The conflict arises due to the different types of QMainWindow and
    BrowserBase.
    """

    pass


# Those are the settings which are stored on each device
# depending on its operating system with QSettings.

qsettings_params = {
    # Antialiasing (works with/without OpenGL, integer because QSettings
    # can't handle booleans)
    'antialiasing': False,
    # Steps per view (relative to time)
    'scroll_sensitivity': 100,
    # Downsampling-Factor (or 'auto', see SettingsDialog for details)
    'downsampling': 1,
    # Downsampling-Method (set SettingsDialog for details)
    'ds_method': 'peak'
}


def _screen_geometry(widget):
    try:
        # Qt 5.14+
        return widget.screen().geometry()
    except AttributeError:
        # Top center of the widget
        screen = QGuiApplication.screenAt(
            widget.mapToGlobal(QPoint(widget.width() // 2, 0)))
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        geometry = screen.geometry()

        return geometry


def _methpartial(meth, **kwargs):
    """Use WeakMethod to create a partial method."""
    meth = weakref.WeakMethod(meth)

    def call(*args_, **kwargs_):
        meth_ = meth()
        if meth_ is not None:
            return meth_(*args_, **kwargs, **kwargs_)

    return call


def _disconnect(sig, *, allow_error=False):
    try:
        sig.disconnect()
    except (TypeError, RuntimeError):  # if there are no connections, ignore it
        if not allow_error:
            raise


class MNEQtBrowser(BrowserBase, QMainWindow, metaclass=_PGMetaClass):
    """A PyQtGraph-backend for 2D data browsing."""

    gotClosed = Signal()

    @_safe_splash
    def __init__(self, **kwargs):
        self.backend_name = 'pyqtgraph'
        self._closed = False

        BrowserBase.__init__(self, **kwargs)
        QMainWindow.__init__(self)

        # Add to list to keep a reference and avoid premature
        # garbage-collection.
        _browser_instances.append(self)

        # Set the browser style
        try:
            from mne.viz.backends._utils import _qt_get_stylesheet
        except Exception:
            stylesheet = None
        else:
            stylesheet = _qt_get_stylesheet(getattr(self.mne, 'theme', 'auto'))
        if stylesheet is not None:
            self.setStyleSheet(stylesheet)

        if self.mne.window_title is not None:
            self.setWindowTitle(self.mne.window_title)
        QApplication.processEvents()  # needs to happen for the theme to be set

        # HiDPI stuff
        self._pixel_ratio = self.devicePixelRatio()
        logger.debug(f'Desktop pixel ratio: {self._pixel_ratio:0.3f}')
        self.mne.mkPen = _methpartial(self._hidpi_mkPen)

        bgcolor = self.palette().color(self.backgroundRole()).getRgbF()[:3]
        self.mne.dark = cspace_convert(bgcolor, 'sRGB1', 'CIELab')[0] < 50

        # update icon theme
        _qt_init_icons()
        if self.mne.dark:
            QIcon.setThemeName('dark')
        else:
            QIcon.setThemeName('light')

        # control raising with _qt_raise_window
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)

        # Initialize attributes which are only used by pyqtgraph, not by
        # matplotlib and add them to MNEBrowseParams.

        # Exactly one MessageBox for messages to facilitate testing/debugging
        self.msg_box = QMessageBox(self)
        # MessageBox modality needs to be adapted for tests
        # (otherwise test execution blocks)
        self.test_mode = False
        # A Settings-Dialog
        self.mne.fig_settings = None
        # Stores decimated data
        self.mne.decim_data = None
        # Stores ypos for selection-mode
        self.mne.selection_ypos_dict = dict()
        # Parameters for precomputing
        self.mne.enable_precompute = False
        self.mne.data_precomputed = False
        self._rerun_load_thread = False
        self.mne.zscore_rgba = None
        # Container for traces
        self.mne.traces = list()
        # Scale-Factor
        self.mne.scale_factor = 1
        # Stores channel-types for butterfly-mode
        self.mne.butterfly_type_order = [tp for tp in
                                         DATA_CH_TYPES_ORDER
                                         if tp in self.mne.ch_types]
        if self.mne.is_epochs:
            # Stores parameters for epochs
            self.mne.epoch_dur = np.diff(self.mne.boundary_times[:2])[0]
            epoch_idx = np.searchsorted(self.mne.midpoints,
                                        (self.mne.t_start,
                                         self.mne.t_start + self.mne.duration))
            self.mne.epoch_idx = np.arange(epoch_idx[0], epoch_idx[1])

        # Load from QSettings if available
        for qparam in qsettings_params:
            default = qsettings_params[qparam]
            qvalue = QSettings().value(qparam, defaultValue=default)
            # QSettings may alter types depending on OS
            if not isinstance(qvalue, type(default)):
                try:
                    qvalue = literal_eval(qvalue)
                except (SyntaxError, ValueError):
                    if qvalue in ['true', 'false']:
                        qvalue = bool(qvalue)
                    else:
                        qvalue = default
            setattr(self.mne, qparam, qvalue)

        # Initialize channel-colors for faster indexing later
        self.mne.ch_color_ref = dict()
        for idx, ch_name in enumerate(self.mne.ch_names):
            ch_type = self.mne.ch_types[idx]
            self.mne.ch_color_ref[ch_name] = self.mne.ch_color_dict[ch_type]

        # Initialize epoch colors for faster indexing later
        if self.mne.is_epochs:
            if self.mne.epoch_colors is None:
                self.mne.epoch_color_ref = \
                    np.repeat([to_rgba_array(c) for c
                               in self.mne.ch_color_ref.values()],
                              len(self.mne.inst), axis=1)
            else:
                self.mne.epoch_color_ref = np.empty((len(self.mne.ch_names),
                                                     len(self.mne.inst), 4))
                for epo_idx, epo in enumerate(self.mne.epoch_colors):
                    for ch_idx, color in enumerate(epo):
                        self.mne.epoch_color_ref[ch_idx, epo_idx] = \
                            to_rgba_array(color)

            # Mark bad epochs
            self.mne.epoch_color_ref[:, self.mne.bad_epochs] = \
                to_rgba_array(self.mne.epoch_color_bad)

            # Mark bad channels
            bad_idxs = np.in1d(self.mne.ch_names, self.mne.info['bads'])
            self.mne.epoch_color_ref[bad_idxs, :] = \
                to_rgba_array(self.mne.ch_color_bad)

        # Add Load-Progressbar for loading in a thread
        self.mne.load_prog_label = QLabel('Loading...')
        self.statusBar().addWidget(self.mne.load_prog_label)
        self.mne.load_prog_label.hide()
        self.mne.load_progressbar = QProgressBar()
        # Set to n_chunks of LoadRunner
        self.mne.load_progressbar.setMaximum(10)
        self.statusBar().addWidget(self.mne.load_progressbar, stretch=1)
        self.mne.load_progressbar.hide()

        # A QThread for preloading
        self.load_thread = LoadThread(self)

        # Create centralWidget and layout
        widget = QWidget()
        layout = QGridLayout()

        # Initialize Axis-Items
        self.mne.time_axis = TimeAxis(self.mne)
        if self.mne.is_epochs:
            self.mne.time_axis.setLabel(text='Epoch Index', units=None)
        else:
            self.mne.time_axis.setLabel(text='Time', units='s')

        self.mne.channel_axis = ChannelAxis(self)
        self.mne.viewbox = RawViewBox(self)

        # Start precomputing if enabled
        self._init_precompute()

        # Parameters for overviewbar
        self.mne.overview_mode = getattr(self.mne, 'overview_mode', 'channels')
        overview_items = dict(
            empty='Empty',
            channels='Channels',
        )
        if self.mne.enable_precompute:
            overview_items['zscore'] = 'Z-Score'
        elif self.mne.overview_mode == 'zscore':
            warn('Cannot use z-score mode without precomputation, setting '
                 'overview_mode="channels"')
            self.mne.overview_mode = 'channels'
        _check_option(
            'overview_mode', self.mne.overview_mode,
            list(overview_items) + ['hidden'])
        hide_overview = False
        if self.mne.overview_mode == 'hidden':
            hide_overview = True
            self.mne.overview_mode = 'channels'

        # Initialize data (needed in DataTrace.update_data).
        self._update_data()

        # Initialize Trace-Plot
        self.mne.plt = PlotItem(viewBox=self.mne.viewbox,
                                axisItems={'bottom': self.mne.time_axis,
                                           'left': self.mne.channel_axis})
        # Hide AutoRange-Button
        self.mne.plt.hideButtons()
        # Configure XY-Range
        if self.mne.is_epochs:
            self.mne.xmax = len(self.mne.inst.times) * len(self.mne.inst) \
                            / self.mne.info['sfreq']
        else:
            self.mne.xmax = self.mne.inst.times[-1]
        # Add one empty line as padding at top (y=0).
        # Negative Y-Axis to display channels from top.
        self.mne.ymax = len(self.mne.ch_order) + 1
        self.mne.plt.setLimits(xMin=0, xMax=self.mne.xmax,
                               yMin=0, yMax=self.mne.ymax)
        # Connect Signals from PlotItem
        self.mne.plt.sigXRangeChanged.connect(self._xrange_changed)
        self.mne.plt.sigYRangeChanged.connect(self._yrange_changed)

        # Add traces
        for ch_idx in self.mne.picks:
            DataTrace(self, ch_idx)

        # Initialize Epochs Grid
        if self.mne.is_epochs:
            grid_pen = self.mne.mkPen(color='k', width=2, style=Qt.DashLine)
            for x_grid in self.mne.boundary_times[1:-1]:
                grid_line = InfiniteLine(pos=x_grid,
                                         pen=grid_pen,
                                         movable=False)
                self.mne.plt.addItem(grid_line)

        # Add events
        if getattr(self.mne, 'event_nums', None) is not None:
            self.mne.events_visible = True
            for ev_time, ev_id in zip(self.mne.event_times,
                                      self.mne.event_nums):
                color = self.mne.event_color_dict[ev_id]
                label = self.mne.event_id_rev.get(ev_id, ev_id)
                event_line = EventLine(self.mne, ev_time, label, color)
                self.mne.event_lines.append(event_line)
        else:
            self.mne.events_visible = False

        # Add Scale-Bars
        self._add_scalebars()

        # Check for OpenGL
        # If a user doesn't specify whether or not to use it:
        # 1. If on macOS, enable it by default to avoid segfault
        # 2. Otherwise, disable it (performance differences seem minimal, and
        #    PyOpenGL is an optional requirement)
        opengl_key = 'MNE_BROWSER_USE_OPENGL'
        if self.mne.use_opengl is None:  # default: opt-in
            # OpenGL needs to be enabled on macOS
            # (https://github.com/mne-tools/mne-qt-browser/issues/53)
            default = 'true' if platform.system() == 'Darwin' else ''
            config_val = get_config(opengl_key, default).lower()
            self.mne.use_opengl = (config_val == 'true')

        if self.mne.use_opengl:
            try:
                import OpenGL
            except (ModuleNotFoundError, ImportError) as exc:
                # On macOS, if use_opengl is True we raise an error because
                # it can lead to segfaults. If a user really knows what they
                # are doing, they can pass use_opengl=False (or set
                # MNE_BROWSER_USE_OPENGL=false)
                if platform.system() == 'Darwin':
                    raise RuntimeError(
                        'Plotting on macOS without OpenGL may be unstable! '
                        'We recommend installing PyOpenGL, but it could not '
                        f'be imported, got:\n{exc}\n\n'
                        'If you want to try plotting without OpenGL, '
                        'you can pass use_opengl=False (use at your own '
                        'risk!). If you know non-OpenGL plotting is stable '
                        'on your system, you can also set the config value '
                        f'{opengl_key}=false to permanently change '
                        'the default behavior on your system.') from None
                # otherwise, emit a warning
                warn('PyOpenGL was not found and OpenGL cannot be used. '
                     'Consider installing pyopengl with pip or conda or set '
                     '"use_opengl=False" to avoid this warning.')
                self.mne.use_opengl = False
            else:
                logger.info(
                    f'Using pyopengl with version {OpenGL.__version__}')
        # Initialize BrowserView (inherits QGraphicsView)
        self.mne.view = BrowserView(self.mne.plt,
                                    useOpenGL=self.mne.use_opengl,
                                    background='w')
        bgcolor = getattr(self.mne, 'bgcolor', 'w')
        self.mne.view.setBackground(_get_color(bgcolor, self.mne.dark))
        layout.addWidget(self.mne.view, 0, 0)

        # Initialize Scroll-Bars
        self.mne.ax_hscroll = TimeScrollBar(self.mne)
        layout.addWidget(self.mne.ax_hscroll, 1, 0, 1, 2)

        self.mne.ax_vscroll = ChannelScrollBar(self.mne)
        layout.addWidget(self.mne.ax_vscroll, 0, 1)

        # Initialize VLine
        self.mne.vline = None
        self.mne.vline_visible = False

        # Initialize crosshair (as in pyqtgraph example)
        self.mne.crosshair_enabled = False
        self.mne.crosshair_h = None
        self.mne.crosshair = None
        self.mne.view.sigSceneMouseMoved.connect(self._mouse_moved)

        # Initialize Annotation-Widgets
        self.mne.annotation_mode = False
        if not self.mne.is_epochs:
            self._init_annot_mode()

        # OverviewBar
        self.mne.overview_bar = OverviewBar(self)
        layout.addWidget(self.mne.overview_bar, 2, 0, 1, 2)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Initialize Selection-Dialog
        if getattr(self.mne, 'group_by', None) in ['position', 'selection']:
            self._create_selection_fig()

        # Initialize Projectors-Dialog if show_options=True
        if getattr(self.mne, 'show_options', False):
            self._toggle_proj_fig()

        # Initialize Toolbar
        self.mne.toolbar = self.addToolBar('Tools')
        # tool_button_style = Qt.ToolButtonTextBesideIcon
        tool_button_style = Qt.ToolButtonIconOnly
        self.mne.toolbar.setToolButtonStyle(tool_button_style)

        adecr_time = QAction(
            QIcon.fromTheme("less_time"), '- Time', parent=self)
        adecr_time.triggered.connect(
            _methpartial(self.change_duration, step=-0.2))
        self.mne.toolbar.addAction(adecr_time)
        aincr_time = QAction(
            QIcon.fromTheme("more_time"), '+ Time', parent=self)
        aincr_time.triggered.connect(
            _methpartial(self.change_duration, step=0.25))
        self.mne.toolbar.addAction(aincr_time)
        self.mne.toolbar.addSeparator()

        adecr_nchan = QAction(
            QIcon.fromTheme("less_channels"), '- Channels', parent=self)
        adecr_nchan.triggered.connect(
            _methpartial(self.change_nchan, step=-10))
        self.mne.toolbar.addAction(adecr_nchan)
        aincr_nchan = QAction(
            QIcon.fromTheme("more_channels"), '+ Channels', parent=self)
        aincr_nchan.triggered.connect(
            _methpartial(self.change_nchan, step=10))
        self.mne.toolbar.addAction(aincr_nchan)
        self.mne.toolbar.addSeparator()

        adecr_nchan = QAction(
            QIcon.fromTheme("zoom_out"), 'Zoom out', parent=self)
        adecr_nchan.triggered.connect(
            _methpartial(self.scale_all, step=4 / 5))
        self.mne.toolbar.addAction(adecr_nchan)
        aincr_nchan = QAction(
            QIcon.fromTheme("zoom_in"), 'Zoom in', parent=self)
        aincr_nchan.triggered.connect(
            _methpartial(self.scale_all, step=5 / 4))
        self.mne.toolbar.addAction(aincr_nchan)
        self.mne.toolbar.addSeparator()

        if not self.mne.is_epochs:
            atoggle_annot = QAction(
                QIcon.fromTheme("annotations"), 'Annotations', parent=self)
            atoggle_annot.triggered.connect(self._toggle_annotation_fig)
            self.mne.toolbar.addAction(atoggle_annot)

        atoggle_proj = QAction(
            QIcon.fromTheme("ssp"), 'SSP', parent=self)
        atoggle_proj.triggered.connect(self._toggle_proj_fig)
        self.mne.toolbar.addAction(atoggle_proj)

        button = QToolButton(self.mne.toolbar)
        button.setToolTip(
            '<h2>Overview-Modes</h2>'
            '<ul>'
            '<li>empty:<br>'
            'Display no background.</li>'
            '<li>channels:<br>'
            'Display each channel with its channel-type color.</li>'
            '<li>zscore:<br>'
            'Display the zscore for the data from each channel across time. '
            'Red indicates high zscores, blue indicates low zscores, '
            'and the boundaries of the color gradient are defined by the '
            'minimum/maximum zscore.'
            'This only works if precompute is set to "True", or if it is '
            'enabled with "auto" and enough free RAM is available.</li>'
            )
        button.setText('Overview Bar')
        button.setIcon(QIcon.fromTheme('overview_bar'))
        button.setToolButtonStyle(tool_button_style)
        menu = self.mne.overview_menu = QMenu(button)
        group = QActionGroup(menu)
        for key, text in overview_items.items():
            radio = QRadioButton(menu)
            radio.setText(text)
            if key == self.mne.overview_mode:
                radio.setChecked(True)
            action = QWidgetAction(menu)
            action.setDefaultWidget(radio)
            menu.addAction(action)
            group.addAction(action)
            radio.clicked.connect(
                _methpartial(
                    self._overview_radio_clicked, menu=menu, new_mode=key))
        menu.addSeparator()
        visible = QAction('Visible', parent=menu)
        menu.addAction(visible)
        visible.setCheckable(True)
        visible.setChecked(True)
        self.mne.overview_bar.setVisible(True)
        visible.triggered.connect(self._toggle_overview_bar)
        if hide_overview:
            # This doesn't work because it hasn't been shown yet:
            # self._toggle_overview_bar()
            visible.setChecked(False)
            self.mne.overview_bar.setVisible(False)
        button.setMenu(self.mne.overview_menu)
        button.setPopupMode(QToolButton.InstantPopup)
        self.mne.toolbar.addWidget(button)

        self.mne.toolbar.addSeparator()

        asettings = QAction(QIcon.fromTheme("settings"), 'Settings',
                            parent=self)
        asettings.triggered.connect(self._toggle_settings_fig)
        self.mne.toolbar.addAction(asettings)

        ahelp = QAction(QIcon.fromTheme("help"), 'Help', parent=self)
        ahelp.triggered.connect(self._toggle_help_fig)
        self.mne.toolbar.addAction(ahelp)

        # Set Start-Range (after all necessary elements are initialized)
        self.mne.plt.setXRange(self.mne.t_start,
                               self.mne.t_start + self.mne.duration,
                               padding=0)
        if self.mne.butterfly:
            self._set_butterfly(True)
        else:
            self.mne.plt.setYRange(0, self.mne.n_channels + 1, padding=0)

        # Set Size
        width = int(self.mne.figsize[0] * self.logicalDpiX())
        height = int(self.mne.figsize[1] * self.logicalDpiY())
        self.resize(width, height)

        # Initialize Keyboard-Shortcuts
        is_mac = platform.system() == 'Darwin'
        dur_keys = ('fn + ←', 'fn + →') if is_mac else ('Home', 'End')
        ch_keys = ('fn + ↑', 'fn + ↓') if is_mac else ('Page up', 'Page down')
        hscroll_type = '1 epoch' if self.mne.is_epochs else '¼ page'

        self.mne.keyboard_shortcuts = {
            'left': {
                'alias': '←',
                'qt_key': Qt.Key_Left,
                'modifier': [None, 'Shift'],
                'slot': [self.hscroll],
                'parameter': ['left', '-full'],
                'description': [f'Scroll left ({hscroll_type})',
                                'Scroll left (full page)']
            },
            'right': {
                'alias': '→',
                'qt_key': Qt.Key_Right,
                'modifier': [None, 'Shift'],
                'slot': [self.hscroll],
                'parameter': ['right', '+full'],
                'description': [f'Scroll right ({hscroll_type})',
                                'Scroll right (full page)']
            },
            'up': {
                'alias': '↑',
                'qt_key': Qt.Key_Up,
                'slot': [self.vscroll],
                'parameter': ['-full'],
                'description': ['Scroll up (full page)']
            },
            'down': {
                'alias': '↓',
                'qt_key': Qt.Key_Down,
                'slot': [self.vscroll],
                'parameter': ['+full'],
                'description': ['Scroll down (full page)']
            },
            'home': {
                'alias': dur_keys[0],
                'qt_key': Qt.Key_Home,
                'kw': 'step',
                'slot': [self.change_duration],
                'parameter': [-0.2],
                'description': [f'Decrease duration ({hscroll_type})']
            },
            'end': {
                'alias': dur_keys[1],
                'qt_key': Qt.Key_End,
                'kw': 'step',
                'slot': [self.change_duration],
                'parameter': [0.25],
                'description': [f'Increase duration ({hscroll_type})']
            },
            'pagedown': {
                'alias': ch_keys[0],
                'qt_key': Qt.Key_PageDown,
                'modifier': [None, 'Shift'],
                'kw': 'step',
                'slot': [self.change_nchan],
                'parameter': [-1, -10],
                'description': ['Decrease shown channels (1)',
                                'Decrease shown channels (10)']
            },
            'pageup': {
                'alias': ch_keys[1],
                'qt_key': Qt.Key_PageUp,
                'modifier': [None, 'Shift'],
                'kw': 'step',
                'slot': [self.change_nchan],
                'parameter': [1, 10],
                'description': ['Increase shown channels (1)',
                                'Increase shown channels (10)']
            },
            '-': {
                'qt_key': Qt.Key_Minus,
                'slot': [self.scale_all],
                'kw': 'step',
                'parameter': [4 / 5],
                'description': ['Decrease Scale']
            },
            '+': {
                'qt_key': Qt.Key_Plus,
                'slot': [self.scale_all],
                'kw': 'step',
                'parameter': [5 / 4],
                'description': ['Increase Scale']
            },
            '=': {
                'qt_key': Qt.Key_Equal,
                'slot': [self.scale_all],
                'kw': 'step',
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
                'slot': [self._check_close],
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
        if self.mne.is_epochs:
            # Disable time format toggling
            del self.mne.keyboard_shortcuts['t']
        else:
            if self.mne.info["meas_date"] is None:
                del self.mne.keyboard_shortcuts["t"]
            # disable histogram of epoch PTP amplitude
            del self.mne.keyboard_shortcuts["h"]

    def _hidpi_mkPen(self, *args, **kwargs):
        kwargs['width'] = self._pixel_ratio * kwargs.get('width', 1.)
        return mkPen(*args, **kwargs)

    def _update_yaxis_labels(self):
        self.mne.channel_axis.repaint()

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

        self._set_scalebars_visible(self.mne.scalebars_visible)

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
        self.mne.overview_bar.set_background()
        if not self.mne.overview_bar.isVisible():
            self._toggle_overview_bar()

    def _overview_radio_clicked(self, checked, *, menu, new_mode):
        menu.close()
        self._overview_mode_changed(new_mode=new_mode)

    def scale_all(self, checked=False, *, step):
        """Scale all traces by multiplying with step."""
        self.mne.scale_factor *= step

        # Reapply clipping if necessary
        if self.mne.clipping is not None:
            self._update_data()

        # Scale Traces (by scaling the Item, not the data)
        for line in self.mne.traces:
            line.update_scale()

        # Update Scalebars
        self._update_scalebar_values()

    def hscroll(self, step):
        """Scroll horizontally by step."""
        if isinstance(step, str):
            if step in ('-full', '+full'):
                rel_step = self.mne.duration
                if step == '-full':
                    rel_step = rel_step * -1
            else:
                assert step in ('left', 'right')
                if self.mne.is_epochs:
                    rel_step = self.mne.duration / self.mne.n_epochs
                else:
                    rel_step = 0.25 * self.mne.duration
                if step == 'left':
                    rel_step = rel_step * -1
        else:
            if self.mne.is_epochs:
                rel_step = (
                    np.sign(step) * self.mne.duration / self.mne.n_epochs
                )
            else:
                rel_step = (
                    step * self.mne.duration / self.mne.scroll_sensitivity
                )
        del step

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
        elif self.mne.butterfly:
            return
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

    def change_duration(self, checked=False, *, step):
        """Change duration by step."""
        xmin, xmax = self.mne.viewbox.viewRange()[0]

        if self.mne.is_epochs:
            # use the length of one epoch as duration change
            min_dur = len(self.mne.inst.times) / self.mne.info['sfreq']
            step_dir = (1 if step > 0 else -1)
            rel_step = min_dur * step_dir
            self.mne.n_epochs = np.clip(self.mne.n_epochs + step_dir,
                                        1, len(self.mne.inst))
        else:
            # never show fewer than 3 samples
            min_dur = 3 * np.diff(self.mne.inst.times[:2])[0]
            rel_step = self.mne.duration * step

        xmax += rel_step

        if xmax - xmin < min_dur:
            xmax = xmin + min_dur

        if xmax > self.mne.xmax:
            diff = xmax - self.mne.xmax
            xmax = self.mne.xmax
            xmin -= diff

        if xmin < 0:
            xmin = 0

        self.mne.ax_hscroll.update_duration()
        self.mne.plt.setXRange(xmin, xmax, padding=0)

    def change_nchan(self, checked=False, *, step):
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

            self.mne.ax_vscroll.update_nchan()
            self.mne.plt.setYRange(ymin, ymax, padding=0)

    def _remove_vline(self):
        if self.mne.vline is not None:
            if self.mne.is_epochs:
                for vline in self.mne.vline:
                    self.mne.plt.removeItem(vline)
            else:
                self.mne.plt.removeItem(self.mne.vline)

        self.mne.vline = None
        self.mne.vline_visible = False
        self.mne.overview_bar.update_vline()

    def _get_vline_times(self, t):
        rel_time = t % self.mne.epoch_dur
        abs_time = self.mne.times[0]
        ts = np.arange(
                self.mne.n_epochs) * self.mne.epoch_dur + abs_time + rel_time

        return ts

    def _vline_slot(self, orig_vline):
        if self.mne.is_epochs:
            ts = self._get_vline_times(orig_vline.value())
            for vl, xt in zip(self.mne.vline, ts):
                if vl != orig_vline:
                    vl.setPos(xt)
        self.mne.overview_bar.update_vline()

    def _add_vline(self, t):
        if self.mne.is_epochs:
            ts = self._get_vline_times(t)

            # Add vline if None
            if self.mne.vline is None:
                self.mne.vline = list()
                for xt in ts:
                    epo_idx = np.clip(
                            np.searchsorted(self.mne.boundary_times, xt) - 1,
                            0, len(self.mne.inst))
                    bmin, bmax = self.mne.boundary_times[epo_idx:epo_idx + 2]
                    # Avoid off-by-one-error at bmax for VlineLabel
                    bmax -= 1 / self.mne.info['sfreq']
                    vl = VLine(self.mne, xt, bounds=(bmin, bmax))
                    # Should only be emitted when dragged
                    vl.sigPositionChangeFinished.connect(self._vline_slot)
                    self.mne.vline.append(vl)
                    self.mne.plt.addItem(vl)
            else:
                for vl, xt in zip(self.mne.vline, ts):
                    vl.setPos(xt)
        else:
            if self.mne.vline is None:
                self.mne.vline = VLine(self.mne, t, bounds=(0, self.mne.xmax))
                self.mne.vline.sigPositionChangeFinished.connect(
                        self._vline_slot)
                self.mne.plt.addItem(self.mne.vline)
            else:
                self.mne.vline.setPos(t)

        self.mne.vline_visible = True
        self.mne.overview_bar.update_vline()

    def _mouse_moved(self, pos):
        """Show Crosshair if enabled at mouse move."""
        if self.mne.crosshair_enabled:
            if self.mne.plt.sceneBoundingRect().contains(pos):
                mousePoint = self.mne.viewbox.mapSceneToView(pos)
                x, y = mousePoint.x(), mousePoint.y()
                if (0 <= x <= self.mne.xmax and
                        0 <= y <= self.mne.ymax):
                    if not self.mne.crosshair:
                        self.mne.crosshair = Crosshair(self.mne)
                        self.mne.plt.addItem(self.mne.crosshair,
                                             ignoreBounds=True)

                    # Get ypos from trace
                    trace = [tr for tr in self.mne.traces if
                             tr.ypos - 0.5 < y < tr.ypos + 0.5]
                    if len(trace) == 1:
                        trace = trace[0]
                        idx = np.searchsorted(self.mne.times, x)
                        if self.mne.data_precomputed:
                            data = self.mne.data[trace.order_idx]
                        else:
                            data = self.mne.data[trace.range_idx]
                        yvalue = data[idx]
                        yshown = yvalue + trace.ypos
                        self.mne.crosshair.set_data(x, yshown)

                        # relative x for epochs
                        if self.mne.is_epochs:
                            rel_idx = idx % len(self.mne.inst.times)
                            x = self.mne.inst.times[rel_idx]

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
        if self.mne.is_epochs:
            if self.mne.vline is not None:
                rel_vl_t = self.mne.vline[0].value() \
                           - self.mne.boundary_times[self.mne.epoch_idx][0]

            # Depends on only allowing xrange showing full epochs
            boundary_idxs = np.searchsorted(self.mne.midpoints, xrange)
            self.mne.epoch_idx = np.arange(*boundary_idxs)

            # Update colors
            for trace in self.mne.traces:
                trace.update_color()

            # Update vlines
            if self.mne.vline is not None:
                for bmin, bmax, vl in zip(self.mne.boundary_times[
                                              self.mne.epoch_idx],
                                          self.mne.boundary_times[
                                              self.mne.epoch_idx + 1],
                                          self.mne.vline):
                    # Avoid off-by-one-error at bmax for VlineLabel
                    bmax -= 1 / self.mne.info['sfreq']
                    vl.setBounds((bmin, bmax))
                    vl.setValue(bmin + rel_vl_t)

        self.mne.t_start = xrange[0]
        self.mne.duration = xrange[1] - xrange[0]

        self._redraw(update_data=True)

        # Update Time-Bar
        self.mne.ax_hscroll.update_value(xrange[0])

        # Update Overview-Bar
        self.mne.overview_bar.update_viewrange()

        # Update Scalebars
        self._update_scalebar_x_positions()

        # Update annotations
        self._update_regions_visible()

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
                trace.remove()
                off_traces.remove(trace)

        # Add new traces if necessary.
        if trace_diff > 0:
            # Make copy to avoid skipping iteration.
            idxs_copy = add_idxs.copy()
            for aidx in idxs_copy[:trace_diff]:
                DataTrace(self, aidx)
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

        if self._rerun_load_thread:
            self._rerun_load_thread = False
            self._init_precompute()

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
            self.load_thread.start()

    def _rerun_precompute(self):
        if self.load_thread.isRunning():
            self._rerun_load_thread = True
        else:
            self._init_precompute()

    def _check_space_for_precompute(self):
        try:
            import psutil
        except ImportError:
            logger.info('Free RAM space could not be determined because'
                        '"psutil" is not installed. '
                        'Setting precompute to False.')
            return False
        else:
            if self.mne.is_epochs:
                files = [self.mne.inst.filename]
            else:
                files = self.mne.inst.filenames
            if files[0] is not None:
                # Get disk-space of raw-file(s)
                disk_space = 0
                for fn in files:
                    disk_space += getsize(fn)

                # Determine expected RAM space based on orig_format
                fmt_multipliers = {'double': 1,
                                   'single': 2,
                                   'int': 2,
                                   'short': 4}

                # Epochs and ICA don't have this attribute, assume single
                # on disk
                fmt = getattr(self.mne.inst, 'orig_format', 'single')
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
                self.mne.data = (
                    self.mne.data - np.nanmean(self.mne.data, axis=1, keepdims=True)
                )
        else:
            # While data is not precomputed get data only from shown range and
            # process only those.
            super()._update_data()

        # Initialize decim
        self.mne.decim_data = np.ones_like(self.mne.picks)
        data_picks_mask = np.in1d(self.mne.picks, self.mne.picks_data)
        self.mne.decim_data[data_picks_mask] = self.mne.decim

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
        screen_geometry = _screen_geometry(self)
        if screen_geometry is None:
            max_pixel_width = 3840  # default=UHD
        else:
            max_pixel_width = screen_geometry.width()
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
    def _add_region(self, plot_onset, duration, description, *, region=None):
        if not region:
            region = AnnotRegion(self.mne, description=description,
                                 values=(plot_onset, plot_onset + duration))
        # Add region to list and plot
        self.mne.regions.append(region)

        # Connect signals of region
        region.regionChangeFinished.connect(self._region_changed)
        region.gotSelected.connect(self._region_selected)
        region.removeRequested.connect(self._remove_region)
        self.mne.viewbox.sigYRangeChanged.connect(region.update_label_pos)
        region.update_label_pos()
        return region

    def _remove_region(self, region, from_annot=True):
        # Remove from shown regions
        if region.label_item in self.mne.plt.items:
            self.mne.plt.removeItem(region.label_item)
        if region in self.mne.plt.items:
            self.mne.plt.removeItem(region)

        # Remove from all regions
        if region in self.mne.regions:
            self.mne.regions.remove(region)

        # Reset selected region
        if region == self.mne.selected_region:
            self.mne.selected_region = None
            # disable, reset start/stop doubleSpinBox until another region is selected
            self.mne.fig_annotation.start_bx.setEnabled(False)
            self.mne.fig_annotation.stop_bx.setEnabled(False)
            with SignalBlocker(self.mne.fig_annotation.start_bx):
                self.mne.fig_annotation.start_bx.setValue(0)
            with SignalBlocker(self.mne.fig_annotation.stop_bx):
                self.mne.fig_annotation.stop_bx.setValue(1 / self.mne.info["sfreq"])

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

    def _init_annot_mode(self):
        self.mne.annotations_visible = True
        self.mne.new_annotation_labels = self._get_annotation_labels()
        if len(self.mne.new_annotation_labels) > 0:
            self.mne.current_description = self.mne.new_annotation_labels[0]
        else:
            self.mne.current_description = None
        self._setup_annotation_colors()
        self.mne.regions = list()
        self.mne.selected_region = None

        # Initialize Annotation-Dock
        existing_dock = getattr(self.mne, 'fig_annotation', None)
        if existing_dock is None:
            self.mne.fig_annotation = AnnotationDock(self)
            self.addDockWidget(Qt.TopDockWidgetArea, self.mne.fig_annotation)
            self.mne.fig_annotation.setVisible(False)
            self.mne.fig_annotation.start_bx.setEnabled(False)
            self.mne.fig_annotation.stop_bx.setEnabled(False)

        # Add annotations as regions
        for annot in self.mne.inst.annotations:
            plot_onset = _sync_onset(self.mne.inst, annot['onset'])
            duration = annot['duration']
            description = annot['description']
            region = self._add_region(plot_onset, duration, description)
            region.update_visible(False)

        # Initialize showing annotation widgets
        self._change_annot_mode()

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
                region.setZValue(
                    2 if region.description == self.mne.current_description else 1
                )
            else:
                region.setZValue(0)

        # Add/Remove selection-rectangle.
        if self.mne.selected_region:
            self.mne.selected_region.select(self.mne.annotation_mode)

    def _toggle_annotation_fig(self):
        if not self.mne.is_epochs:
            self.mne.annotation_mode = not self.mne.annotation_mode
            self._change_annot_mode()

    def _update_regions_visible(self):
        if self.mne.is_epochs:
            return
        start = self.mne.t_start
        stop = start + self.mne.duration
        for region in self.mne.regions:
            if self.mne.visible_annotations[region.description]:
                rgn = region.getRegion()
                # Avoid NumPy bool here
                visible = bool(rgn[0] <= stop and rgn[1] >= start)
            else:
                visible = False
            region.update_visible(visible)
        self.mne.overview_bar.update_annotations()

    def _set_annotations_visible(self, visible):
        for descr in self.mne.visible_annotations:
            self.mne.visible_annotations[descr] = visible
        self._update_regions_visible()

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
        self._rerun_precompute()
        self._redraw()

    def _toggle_proj_fig(self):
        if self.mne.fig_proj is None:
            ProjDialog(self, name='fig_proj')
        else:
            self.mne.fig_proj.close()
            self.mne.fig_proj = None

    def _toggle_all_projs(self):
        if self.mne.fig_proj is None:
            self._apply_update_projectors(toggle_all=True)
        else:
            self.mne.fig_proj.toggle_all()

    def _toggle_whitening(self):
        if self.mne.noise_cov is not None:
            super()._toggle_whitening()
            # If data was precomputed it needs to be precomputed again.
            self._rerun_precompute()
            self._redraw()

    def _toggle_settings_fig(self):
        if self.mne.fig_settings is None:
            SettingsDialog(self, name='fig_settings')
        else:
            self.mne.fig_settings.close()
            self.mne.fig_settings = None

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
            self.mne.ymax = ymax
            self.mne.plt.setLimits(yMax=ymax)
            self.mne.plt.setYRange(0, ymax, padding=0)
        elif butterfly:
            ymax = len(self.mne.butterfly_type_order) + 1
            self.mne.ymax = ymax
            self.mne.plt.setLimits(yMax=ymax)
            self.mne.plt.setYRange(0, ymax, padding=0)
        else:
            self.mne.ymax = len(self.mne.ch_order) + 1
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
            trace.update_color()
            trace.update_ypos()

        self._draw_traces()

    def _toggle_butterfly(self):
        if self.mne.instance_type != 'ica':
            self._set_butterfly(not self.mne.butterfly)

    def _toggle_dc(self):
        self.mne.remove_dc = not self.mne.remove_dc
        self._redraw()

    def _set_events_visible(self, visible):
        for event_line in self.mne.event_lines:
            event_line.setVisible(visible)

        self.mne.overview_bar.update_events()

    def _toggle_events(self):
        if self.mne.event_nums is not None:
            self.mne.events_visible = not self.mne.events_visible
            self._set_events_visible(self.mne.events_visible)

    def _toggle_time_format(self):
        if self.mne.info["meas_date"] is not None:
            if self.mne.time_format == 'float':
                self.mne.time_format = 'clock'
                self.mne.time_axis.setLabel(text='Time of day')
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
        visible = not self.mne.overview_bar.isVisible()
        for item in self.mne.overview_menu.actions():
            if item.text() == 'Visible':
                item.setChecked(visible)
                break
        self.mne.overview_bar.setVisible(visible)

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
        canvas.setFocusPolicy(Qt.FocusPolicy(Qt.StrongFocus | Qt.WheelFocus))
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

    def _toggle_epoch_histogram(self):
        if self.mne.is_epochs:
            fig = self._create_epoch_histogram()
            if fig is not None:
                self._get_dlg_from_mpl(fig)

    def _create_selection_fig(self):
        if not any([isinstance(fig, SelectionDialog) for
                    fig in self.mne.child_figs]):
            SelectionDialog(self)

    def message_box(self, text, info_text=None, buttons=None,
                    default_button=None, icon=None, modal=True):  # noqa: D102
        self.msg_box.setText(f'<font size="+2"><b>{text}</b></font>')
        if info_text is not None:
            self.msg_box.setInformativeText(info_text)
        if buttons is not None:
            self.msg_box.setStandardButtons(buttons)
        if default_button is not None:
            self.msg_box.setDefaultButton(default_button)
        if icon is not None:
            self.msg_box.setIcon(icon)

        # Allow interacting with message_box in test-mode.
        # Set modal=False only if no return value is expected.
        self.msg_box.setModal(False if self.test_mode else modal)
        if self.test_mode or not modal:
            self.msg_box.show()
        else:
            return self.msg_box.exec()

    def keyPressEvent(self, event):
        """Customize key press events."""
        # On MacOs additionally KeypadModifier is set when arrow-keys
        # are pressed.
        mods = event.modifiers()
        try:
            mods = int(mods)  # PyQt < 5.13
        except Exception:
            pass
        modifiers = {
            'Shift': bool(Qt.ShiftModifier & mods),
            'Ctrl': bool(Qt.ControlModifier & mods),
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
                    val = key_dict['parameter'][param_idx]
                    if 'kw' in key_dict:
                        slot(**{key_dict['kw']: val})
                    else:
                        slot(val)
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
        elif key.startswith('shift+'):
            key = key[6:]
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
        self.mne.channel_axis.repaint()
        # Wait because channel-axis may need time
        # (came up with test_epochs::test_plot_epochs_clicks)
        QTest.qWait(100)
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

    def show(self):  # noqa: D102
        # Set raise_window like matplotlib if possible
        super().show()
        _qt_raise_window(self)

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

    def _check_close(self):
        """Close annotations-mode before closing the browser."""
        if self.mne.annotation_mode:
            self._toggle_annotation_fig()
        else:
            self.close()

    def closeEvent(self, event):
        """Customize close event."""
        event.accept()
        if hasattr(self, 'mne'):
            # Explicit disconnects to avoid reference cycles that gc can't
            # properly resolve ()
            if hasattr(self.mne, 'plt'):
                _disconnect(self.mne.plt.sigXRangeChanged)
                _disconnect(self.mne.plt.sigYRangeChanged)
            if hasattr(self.mne, 'toolbar'):
                for action in self.mne.toolbar.actions():
                    allow_error = action.text() == ''
                    _disconnect(action.triggered, allow_error=allow_error)
            # Save settings going into QSettings.
            for qsetting in qsettings_params:
                value = getattr(self.mne, qsetting)
                QSettings().setValue(qsetting, value)
            for attr in ('keyboard_shortcuts', 'traces', 'plt', 'toolbar',
                         'fig_annotation'):
                if hasattr(self.mne, attr):
                    delattr(self.mne, attr)
            if hasattr(self.mne, 'child_figs'):
                for fig in self.mne.child_figs:
                    fig.close()
                self.mne.child_figs.clear()
            for attr in ('traces', 'event_lines', 'regions'):
                getattr(self.mne, attr, []).clear()
            if getattr(self.mne, 'vline', None) is not None:
                if self.mne.is_epochs:
                    for vl in self.mne.vline:
                        _disconnect(vl.sigPositionChangeFinished)
                    self.mne.vline.clear()
                else:
                    _disconnect(self.mne.vline.sigPositionChangeFinished)
        if getattr(self, 'load_thread', None) is not None:
            self.load_thread.clean()
            self.load_thread = None

        # Remove self from browser_instances in globals
        if self in _browser_instances:
            _browser_instances.remove(self)
        self._close(event)
        self.gotClosed.emit()
        # Make sure it gets deleted after it was closed.
        self.deleteLater()
        self._closed = True

    def _fake_click_on_toolbar_action(self, action_name, wait_after=500):
        """Trigger event associated with action 'action_name' in toolbar."""
        for action in self.mne.toolbar.actions():
            if not action.isSeparator():
                if action.iconText() == action_name:
                    action.trigger()
        QTest.qWait(wait_after)


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


# modified from: https://github.com/pyvista/pyvistaqt
def _setup_ipython(ipython=None):
    # ipython magic
    if scooby.in_ipython():
        from IPython import get_ipython
        ipython = get_ipython()
        ipython.run_line_magic("gui", "qt")
        from IPython.external.qt_for_kernel import QtGui
        QtGui.QApplication.instance()
    return ipython


def _qt_init_icons():
    from qtpy.QtGui import QIcon
    icons_path = f"{Path(__file__).parent}/icons"
    QIcon.setThemeSearchPaths([icons_path])
    return icons_path


def _init_browser(**kwargs):
    _setup_ipython()
    setConfigOption('enableExperimental', True)
    app_kwargs = dict()
    if kwargs.get('splash', False):
        app_kwargs['splash'] = 'Initializing mne-qt-browser...'
    out = _init_mne_qtapp(pg_app=True, **app_kwargs)
    if 'splash' in app_kwargs:
        kwargs['splash'] = out[1]  # returned as second element
    browser = MNEQtBrowser(**kwargs)

    return browser


class PyQtGraphBrowser(MNEQtBrowser):  # noqa: D101
    pass  # just for backward compat with MNE 1.0 scraping


class SignalBlocker(QSignalBlocker):
    """Wrapper to use QSignalBlocker as a context manager in PySide2."""

    def __enter__(self):
        if hasattr(super(), "__enter__"):
            super().__enter__()
        else:
            super().reblock()

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(super(), "__exit__"):
            super().__exit__(exc_type, exc_value, traceback)
        else:
            super().unblock()
