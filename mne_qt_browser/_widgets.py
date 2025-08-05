import datetime
import platform
import weakref
from collections import OrderedDict

import numpy as np
from mne.annotations import _sync_onset
from mne.viz.utils import _merge_annotations
from pyqtgraph import AxisItem, GraphicsView, Point, ViewBox, mkBrush
from qtpy.QtCore import QLineF, QPoint, QPointF, QRectF, QSignalBlocker, Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QMessageBox,
    QScrollBar,
    QStyle,
    QStyleOptionSlider,
)

from mne_qt_browser._colors import _get_color
from mne_qt_browser._graphic_items import AnnotRegion
from mne_qt_browser._utils import DATA_CH_TYPES_ORDER, _screen_geometry


def _mouse_event_position(ev):
    try:  # Qt6
        return ev.position()
    except AttributeError:
        return ev.pos()


class BrowserView(GraphicsView):
    """Customized View as part of GraphicsView framework."""

    def __init__(self, plot, **kwargs):
        super().__init__(**kwargs)
        self.setCentralItem(plot)
        self.viewport().setAttribute(Qt.WA_AcceptTouchEvents, True)

        self.viewport().grabGesture(Qt.PinchGesture)
        self.viewport().grabGesture(Qt.SwipeGesture)

    def mouseMoveEvent(self, ev):
        """Customize MouseMoveEvent."""
        # Don't set GraphicsView.mouseEnabled to True, we only want part of the
        # functionality PyQtGraph offers here.
        super().mouseMoveEvent(ev)
        self.sigSceneMouseMoved.emit(_mouse_event_position(ev))


class ChannelAxis(AxisItem):
    """The y-axis displaying the channel names."""

    def __init__(self, main):
        self.weakmain = weakref.ref(main)
        self.mne = main.mne
        del main
        self.ch_texts = OrderedDict()
        super().__init__(orientation="left")
        self.style["autoReduceTextSpace"] = False

    def tickValues(self, minVal, maxVal, size):
        """Customize creation of axis values from visible axis range."""
        minVal, maxVal = sorted((minVal, maxVal))
        values = list(range(round(minVal) + 1, round(maxVal)))
        tick_values = [(1, values)]
        return tick_values

    def tickStrings(self, values, scale, spacing):
        """Customize strings of axis values."""
        # Get channel names
        if self.mne.butterfly and self.mne.fig_selection is not None:
            tick_strings = list(self.weakmain()._make_butterfly_selections_dict())
        elif self.mne.butterfly:
            _, ixs, _ = np.intersect1d(
                DATA_CH_TYPES_ORDER, self.mne.ch_types, return_indices=True
            )
            ixs.sort()
            tick_strings = np.array(DATA_CH_TYPES_ORDER)[ixs]
        else:
            # Get channel names and by substracting 1 from tick values since the first
            # channel starts at y=1
            tick_strings = self.mne.ch_names[self.mne.ch_order[[v - 1 for v in values]]]

        return tick_strings

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        """Customize drawing of axis items."""
        super().drawPicture(p, axisSpec, tickSpecs, textSpecs)
        for rect, flags, text in textSpecs:
            if self.mne.butterfly and self.mne.fig_selection is not None:
                p.setPen(_get_color("black", self.mne.dark))
            elif self.mne.butterfly:
                p.setPen(_get_color(self.mne.ch_color_dict[text], self.mne.dark))
            elif text in self.mne.info["bads"]:
                p.setPen(_get_color(self.mne.ch_color_bad, self.mne.dark))
            else:
                p.setPen(_get_color(self.mne.ch_color_ref[text], self.mne.dark))
            self.ch_texts[text] = (
                (rect.left(), rect.left() + rect.width()),
                (rect.top(), rect.top() + rect.height()),
            )
            p.drawText(rect, int(flags), text)

    def repaint(self):
        """Repaint Channel Axis."""
        self.picture = None
        self.update()

    def mouseClickEvent(self, event):
        """Customize mouse click events."""
        # Clean up channel texts
        if not self.mne.butterfly:
            self.ch_texts = {
                k: v
                for k, v in self.ch_texts.items()
                if k in [tr.ch_name for tr in self.mne.traces]
            }
            # Get channel name from position of channel description
            ypos = event.scenePos().y()
            y_values = list(self.ch_texts.values())
            if len(y_values) == 0:
                return
            y_values = np.array(y_values, float)[:, 1, :]
            y_diff = np.abs(y_values - ypos)
            ch_idx = int(np.argmin(y_diff, axis=0)[0])
            ch_name = list(self.ch_texts)[ch_idx]
            trace = [tr for tr in self.mne.traces if tr.ch_name == ch_name][0]
            if event.button() == Qt.LeftButton:
                trace.toggle_bad()
            elif event.button() == Qt.RightButton:
                self.weakmain()._create_ch_context_fig(trace.range_idx)

    def get_labels(self):
        """Get labels for testing."""
        values = self.tickValues(*self.mne.viewbox.viewRange()[1], None)
        labels = self.tickStrings(values[0][1], None, None)

        return labels


class RawViewBox(ViewBox):
    """PyQtGraph wrapper for interaction with the View."""

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

        if event.button() == Qt.LeftButton and self.mne.annotation_mode:
            if self.mne.current_description:
                description = self.mne.current_description
                if event.isStart():
                    self._drag_start = self.mapSceneToView(event.lastScenePos()).x()
                    self._drag_start = 0 if self._drag_start < 0 else self._drag_start
                    drag_stop = self.mapSceneToView(event.scenePos()).x()
                    self._drag_region = AnnotRegion(
                        self.mne,
                        description=description,
                        values=(self._drag_start, drag_stop),
                        weakmain=self.weakmain,
                    )
                elif event.isFinish():
                    drag_stop = self.mapSceneToView(event.scenePos()).x()
                    drag_stop = 0 if drag_stop < 0 else drag_stop
                    xmax = self.mne.xmax + 1 / self.mne.info["sfreq"]
                    drag_stop = xmax if xmax < drag_stop else drag_stop
                    self._drag_region.setRegion((self._drag_start, drag_stop))
                    plot_onset = min(self._drag_start, drag_stop)
                    plot_offset = max(self._drag_start, drag_stop)
                    duration = abs(self._drag_start - drag_stop)

                    # Add to annotations
                    onset = _sync_onset(self.mne.inst, plot_onset, inverse=True)
                    _merge_annotations(
                        onset,
                        onset + duration,
                        self.mne.current_description,
                        self.mne.inst.annotations,
                    )

                    # Add to regions/merge regions
                    merge_values = [plot_onset, plot_offset]
                    rm_regions = list()
                    for region in self.mne.regions:
                        if region.description != self.mne.current_description:
                            continue
                        values = region.getRegion()
                        if any(plot_onset <= val <= plot_offset for val in values):
                            merge_values += values
                            rm_regions.append(region)
                    if len(merge_values) > 2:
                        self._drag_region.setRegion(
                            (min(merge_values), max(merge_values))
                        )
                    for rm_region in rm_regions:
                        self.weakmain()._remove_region(rm_region, from_annot=False)
                    self.weakmain()._add_region(
                        plot_onset,
                        duration,
                        self.mne.current_description,
                        region=self._drag_region,
                    )
                    self._drag_region.select(True)
                    self._drag_region.setZValue(2)

                    # Update Overview Bar
                    self.mne.overview_bar.update_annotations()
                else:
                    x_to = self.mapSceneToView(event.scenePos()).x()
                    with QSignalBlocker(self._drag_region):
                        self._drag_region.setRegion((self._drag_start, x_to))

            elif event.isFinish():
                self.weakmain().message_box(
                    text="No description!",
                    info_text="No description is given, add one!",
                    buttons=QMessageBox.Ok,
                    icon=QMessageBox.Warning,
                )

    def mouseClickEvent(self, event):
        """Customize mouse click events."""
        # If we want the context menu back, uncomment the following line
        # super().mouseClickEvent(event)
        if not self.mne.annotation_mode:
            if event.button() == Qt.LeftButton:
                self.weakmain()._add_vline(self.mapSceneToView(event.scenePos()).x())
            elif event.button() == Qt.RightButton:
                self.weakmain()._remove_vline()

    def wheelEvent(self, ev, axis=None):
        """Customize mouse wheel/trackpad scroll events."""
        ev.accept()
        scroll = -1 * ev.delta() / 120
        if ev.orientation() == Qt.Horizontal:
            self.weakmain().hscroll(scroll * 10)
        elif ev.orientation() == Qt.Vertical:
            self.weakmain().vscroll(scroll)

    def keyPressEvent(self, event):  # noqa: D102
        self.weakmain().keyPressEvent(event)


class TimeAxis(AxisItem):
    """The x-axis displaying the time."""

    def __init__(self, mne):
        self.mne = mne
        self._spacing = None
        super().__init__(orientation="bottom")

    def tickValues(self, minVal, maxVal, size):
        """Customize creation of axis values from visible axis range."""
        if self.mne.is_epochs:
            value_idxs = np.searchsorted(self.mne.midpoints, [minVal, maxVal])
            values = self.mne.midpoints[slice(*value_idxs)]
            spacing = len(self.mne.inst.times) / self.mne.info["sfreq"]
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

        elif self.mne.time_format == "clock":
            meas_date = self.mne.info["meas_date"]
            first_time = datetime.timedelta(seconds=self.mne.inst.first_time)

            digits = np.ceil(-np.log10(min(v[0] for v in self._spacing)) + 1).astype(
                int
            )
            tick_strings = list()
            for val in values:
                val_time = datetime.timedelta(seconds=val) + first_time + meas_date
                val_str = val_time.strftime("%H:%M:%S")
                if int(val_time.microsecond):
                    val_str += f"{round(val_time.microsecond * 1e-6, digits)}"[1:]
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
        values = self.tickValues(*self.mne.viewbox.viewRange()[0], self.mne.xmax)
        labels = list()
        for spacing, vals in values:
            labels += self.tickStrings(vals, 1, spacing)

        return labels


class OverviewBar(QGraphicsView):
    """
    Provides overview over channels and current visible range.

    Has different modes:
    - channels: Display channel types
    - z-score: Display channel-wise z-scores across time
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

        # Initialize graphics items
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
        epoch_line_pen = self.mne.mkPen(color="k", width=1)
        for t in self.mne.boundary_times[1:-1]:
            top_left = self._mapFromData(t, 0)
            bottom_right = self._mapFromData(t, len(self.mne.ch_order))
            line = self.scene().addLine(QLineF(top_left, bottom_right), epoch_line_pen)
            line.setZValue(1)
            self.epoch_line_dict[t] = line

    def update_bad_channels(self):
        """Update representation of bad channels."""
        bad_set = set(self.mne.info["bads"])
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
                start, stop = self.mne.boundary_times[epo_idx : epo_idx + 2]
                top_left = self._mapFromData(start, 0)
                bottom_right = self._mapFromData(stop, len(self.mne.ch_order))
                pen = _get_color(self.mne.epoch_color_bad, self.mne.dark)
                rect = self.scene().addRect(
                    QRectF(top_left, bottom_right), pen=pen, brush=pen
                )
                rect.setZValue(3)
                self.bad_epoch_rect_dict[epo_num] = rect
            elif epo_num in rm_epos:
                self.scene().removeItem(self.bad_epoch_rect_dict[epo_num])
                self.bad_epoch_rect_dict.pop(epo_num)

    def update_events(self):
        """Update representation of events."""
        if (
            getattr(self.mne, "event_nums", None) is not None
            and self.mne.events_visible
        ):
            for ev_t, ev_id in zip(self.mne.event_times, self.mne.event_nums):
                color_name = self.mne.event_color_dict[ev_id]
                color = _get_color(color_name, self.mne.dark)
                color.setAlpha(100)
                pen = self.mne.mkPen(color)
                top_left = self._mapFromData(ev_t, 0)
                bottom_right = self._mapFromData(ev_t, len(self.mne.ch_order))
                line = self.scene().addLine(QLineF(top_left, bottom_right), pen)
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
        annot_set = set(
            [
                annot["onset"]
                for annot in annotations
                if self.mne.visible_annotations[annot["description"]]
            ]
        )
        rect_set = set(self.annotations_rect_dict)

        add_onsets = annot_set.difference(rect_set)
        rm_onsets = rect_set.difference(annot_set)

        # Add missing onsets
        for add_onset in add_onsets:
            plot_onset = _sync_onset(self.mne.inst, add_onset)
            annot_idx = np.argwhere(self.mne.inst.annotations.onset == add_onset)[0][0]
            duration = annotations.duration[annot_idx]
            description = annotations.description[annot_idx]
            color_name = self.mne.annotation_segment_colors[description]
            color = _get_color(color_name, self.mne.dark)
            color.setAlpha(150)
            pen = self.mne.mkPen(color)
            brush = mkBrush(color)
            top_left = self._mapFromData(plot_onset, 0)
            bottom_right = self._mapFromData(
                plot_onset + duration, len(self.mne.ch_order)
            )
            rect = self.scene().addRect(QRectF(top_left, bottom_right), pen, brush)
            rect.setZValue(3)
            self.annotations_rect_dict[add_onset] = {
                "rect": rect,
                "plot_onset": plot_onset,
                "duration": duration,
                "color": color_name,
            }

        # Remove onsets
        for rm_onset in rm_onsets:
            self.scene().removeItem(self.annotations_rect_dict[rm_onset]["rect"])
            self.annotations_rect_dict.pop(rm_onset)

        # Changes
        for edit_onset in self.annotations_rect_dict:
            plot_onset = _sync_onset(self.mne.inst, edit_onset)
            annot_idx = np.where(annotations.onset == edit_onset)[0][0]
            duration = annotations.duration[annot_idx]
            rect_duration = self.annotations_rect_dict[edit_onset]["duration"]
            rect = self.annotations_rect_dict[edit_onset]["rect"]
            # Update changed duration
            if duration != rect_duration:
                self.annotations_rect_dict[edit_onset]["duration"] = duration
                top_left = self._mapFromData(plot_onset, 0)
                bottom_right = self._mapFromData(
                    plot_onset + duration, len(self.mne.ch_order)
                )
                rect.setRect(QRectF(top_left, bottom_right))
            # Update changed color
            description = annotations.description[annot_idx]
            color_name = self.mne.annotation_segment_colors[description]
            rect_color = self.annotations_rect_dict[edit_onset]["color"]
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
            # VLine representation not useful in epochs mode
            pass
        # Add VLine representation
        elif self.mne.vline is not None:
            value = self.mne.vline.value()
            top_left = self._mapFromData(value, 0)
            bottom_right = self._mapFromData(value, len(self.mne.ch_order))
            line = QLineF(top_left, bottom_right)
            if self.v_line is None:
                pen = self.mne.mkPen("g")
                self.v_line = self.scene().addLine(line, pen)
                self.v_line.setZValue(1)
            else:
                self.v_line.setLine(line)
        # Remove VLine representation
        elif self.v_line is not None:
            self.scene().removeItem(self.v_line)
            self.v_line = None

    def update_viewrange(self):
        """Update representation of viewrange."""
        if self.mne.butterfly:
            top_left = self._mapFromData(self.mne.t_start, 0)
            bottom_right = self._mapFromData(
                self.mne.t_start + self.mne.duration, self.mne.ymax
            )
        else:
            top_left = self._mapFromData(self.mne.t_start, self.mne.ch_start)
            bottom_right = self._mapFromData(
                self.mne.t_start + self.mne.duration,
                self.mne.ch_start + self.mne.n_channels,
            )
        rect = QRectF(top_left, bottom_right)
        if self.viewrange_rect is None:
            pen = self.mne.mkPen(color="g")
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
            if x == "-offbounds":
                epo_idx = 0
            elif x == "+offbounds":
                epo_idx = len(self.mne.inst) - self.mne.n_epochs
            else:
                epo_idx = max(x - self.mne.n_epochs // 2, 0)
            x = self.mne.boundary_times[epo_idx]
        elif x == "-offbounds":
            x = 0
        elif x == "+offbounds":
            x = self.mne.xmax - self.mne.duration
        else:
            # Move click position to middle of view range
            x -= self.mne.duration / 2
        xmin = np.clip(x, 0, self.mne.xmax - self.mne.duration)
        xmax = np.clip(xmin + self.mne.duration, self.mne.duration, self.mne.xmax)

        self.mne.plt.setXRange(xmin, xmax, padding=0)

        # Set Y
        if y == "-offbounds":
            y = 0
        elif y == "+offbounds":
            y = self.mne.ymax - (self.mne.n_channels + 1)
        else:
            # Move click position to middle of view range
            y -= self.mne.n_channels / 2
        ymin = np.clip(y, 0, self.mne.ymax - (self.mne.n_channels + 1))
        ymax = np.clip(
            ymin + self.mne.n_channels + 1, self.mne.n_channels, self.mne.ymax
        )
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
        # This temporarily circumvents a bug, which only appears on Windows and when
        # pyqt>=5.14.2 is installed from conda-forge. It leads to receiving
        # mouseMoveEvents all the time when the Mouse is moved through the OverviewBar,
        # even when now MouseBUtton is pressed. Dragging the mouse on OverviewBar is
        # then not possible anymore.
        if not platform.system() == "Windows":
            self._set_range_from_pos(event.pos())

    def _fit_bg_img(self):
        # Remove previous item from scene
        if self.bg_pxmp_item is not None and self.bg_pxmp_item in self.scene().items():
            self.scene().removeItem(self.bg_pxmp_item)
        # Resize Pixmap
        if self.bg_pxmp is not None:
            cnt_rect = self.contentsRect()
            self.bg_pxmp = self.bg_pxmp.scaled(
                cnt_rect.width(), cnt_rect.height(), Qt.IgnoreAspectRatio
            )
            self.bg_pxmp_item = self.scene().addPixmap(self.bg_pxmp)

    def resizeEvent(self, event):
        """Customize resize event."""
        super().resizeEvent(event)
        cnt_rect = self.contentsRect()
        self.setSceneRect(
            QRectF(QPointF(0, 0), QPointF(cnt_rect.width(), cnt_rect.height()))
        )
        # Resize background
        self._fit_bg_img()

        # Resize Graphics Items (assuming height never changes)
        # Resize bad_channels
        for bad_ch_line in self.bad_line_dict.values():
            current_line = bad_ch_line.line()
            bad_ch_line.setLine(
                QLineF(current_line.p1(), Point(cnt_rect.width(), current_line.y2()))
            )

        # Resize event lines
        for ev_t, event_line in self.event_line_dict.items():
            top_left = self._mapFromData(ev_t, 0)
            bottom_right = self._mapFromData(ev_t, len(self.mne.ch_order))
            event_line.setLine(QLineF(top_left, bottom_right))

        if self.mne.is_epochs:
            # Resize epoch lines
            for epo_t, epoch_line in self.epoch_line_dict.items():
                top_left = self._mapFromData(epo_t, 0)
                bottom_right = self._mapFromData(epo_t, len(self.mne.ch_order))
                epoch_line.setLine(QLineF(top_left, bottom_right))
            # Resize bad rects
            for epo_idx, epoch_rect in self.bad_epoch_rect_dict.items():
                start, stop = self.mne.boundary_times[epo_idx : epo_idx + 2]
                top_left = self._mapFromData(start, 0)
                bottom_right = self._mapFromData(stop, len(self.mne.ch_order))
                epoch_rect.setRect(QRectF(top_left, bottom_right))
        else:
            # Resize annotation rects
            for annot_dict in self.annotations_rect_dict.values():
                annot_rect = annot_dict["rect"]
                plot_onset = annot_dict["plot_onset"]
                duration = annot_dict["duration"]

                top_left = self._mapFromData(plot_onset, 0)
                bottom_right = self._mapFromData(
                    plot_onset + duration, len(self.mne.ch_order)
                )
                annot_rect.setRect(QRectF(top_left, bottom_right))

        # Update vline
        if all([i is not None for i in [self.v_line, self.mne.vline]]):
            value = self.mne.vline.value()
            top_left = self._mapFromData(value, 0)
            bottom_right = self._mapFromData(value, len(self.mne.ch_order))
            self.v_line.setLine(QLineF(top_left, bottom_right))

        # Update viewrange rect
        top_left = self._mapFromData(self.mne.t_start, self.mne.ch_start)
        bottom_right = self._mapFromData(
            self.mne.t_start + self.mne.duration,
            self.mne.ch_start + self.mne.n_channels,
        )
        self.viewrange_rect.setRect(QRectF(top_left, bottom_right))

    def set_background(self):
        """Set the background image for the selected overview mode."""
        # Add overview pixmap
        self.bg_pxmp = None
        if self.mne.overview_mode == "empty":
            pass
        elif self.mne.overview_mode == "channels":
            channel_rgba = np.empty((len(self.mne.ch_order), 2, 4))
            for line_idx, ch_idx in enumerate(self.mne.ch_order):
                ch_type = self.mne.ch_types[ch_idx]
                color = _get_color(self.mne.ch_color_dict[ch_type], self.mne.dark)
                channel_rgba[line_idx, :] = color.getRgb()

            channel_rgba = np.require(channel_rgba, np.uint8, "C")
            self.bg_img = QImage(
                channel_rgba,
                channel_rgba.shape[1],
                channel_rgba.shape[0],
                QImage.Format_RGBA8888,
            )
            self.bg_pxmp = QPixmap.fromImage(self.bg_img)

        elif self.mne.overview_mode == "zscore" and self.mne.zscore_rgba is not None:
            self.bg_img = QImage(
                self.mne.zscore_rgba,
                self.mne.zscore_rgba.shape[1],
                self.mne.zscore_rgba.shape[0],
                QImage.Format_RGBA8888,
            )
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
            x = "-offbounds"
        elif xnorm > 1:
            x = "+offbounds"
        else:
            if self.mne.is_epochs:
                # Return epoch index for epochs
                x = int(len(self.mne.inst) * xnorm)
            else:
                time_idx = int((len(self.mne.inst.times) - 1) * xnorm)
                x = self.mne.inst.times[time_idx]

        ynorm = point.y() / self.height()
        if ynorm < 0:
            y = "-offbounds"
        elif ynorm > 1:
            y = "+offbounds"
        else:
            y = len(self.mne.ch_order) * ynorm

        return x, y

    def keyPressEvent(self, event):  # noqa: D102
        self.weakmain().keyPressEvent(event)


class BaseScrollBar(QScrollBar):
    """Base class for scrolling directly to the clicked position."""

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
                QStyle.CC_ScrollBar, opt, pos, self
            )
            if (
                control == QStyle.SC_ScrollBarAddPage
                or control == QStyle.SC_ScrollBarSubPage
            ):
                # scroll here
                gr = self.style().subControlRect(
                    QStyle.CC_ScrollBar, opt, QStyle.SC_ScrollBarGroove, self
                )
                sr = self.style().subControlRect(
                    QStyle.CC_ScrollBar, opt, QStyle.SC_ScrollBarSlider, self
                )
                if self.orientation() == Qt.Horizontal:
                    pos_ = pos.x()
                    sliderLength = sr.width()
                    sliderMin = gr.x()
                    sliderMax = gr.right() - sliderLength + 1
                    if self.layoutDirection() == Qt.RightToLeft:
                        opt.upsideDown = not opt.upsideDown
                else:
                    pos_ = pos.y()
                    sliderLength = sr.height()
                    sliderMin = gr.y()
                    sliderMax = gr.bottom() - sliderLength + 1
                self.setValue(
                    QStyle.sliderValueFromPosition(
                        self.minimum(),
                        self.maximum(),
                        pos_ - sliderMin,
                        sliderMax - sliderMin,
                        opt.upsideDown,
                    )
                )
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
        # Because valueChanged is needed (captures every input to scrollbar, not just
        # sliderMoved), there has to be made a differentiation between internal and
        # external changes.
        self.external_change = False
        self.valueChanged.connect(self._time_changed)

    def _time_changed(self, value):
        if not self.external_change:
            if self.mne.is_epochs:
                # Convert Epoch index to time
                value = self.mne.boundary_times[int(value)]
            else:
                value /= self.step_factor
            self.mne.plt.setXRange(value, value + self.mne.duration, padding=0)

    def update_value(self, value):
        """Update value of the ScrollBar."""
        # Mark change as external to avoid setting XRange again in _time_changed
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
            self.setMaximum(int((self.mne.xmax - self.mne.duration) * self.step_factor))

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
        # Because valueChanged is needed (captures every input to scrollbar, not just
        # sliderMoved), there has to be made a differentiation between internal and
        # external changes
        self.external_change = False
        self.valueChanged.connect(self._channel_changed)

    def _channel_changed(self, value):
        if not self.external_change:
            if self.mne.fig_selection:
                label = list(self.mne.ch_selections)[value]
                self.mne.fig_selection._chkbx_changed(None, label)
            elif not self.mne.butterfly:
                value = min(value, self.mne.ymax - self.mne.n_channels)
                self.mne.plt.setYRange(
                    value, value + self.mne.n_channels + 1, padding=0
                )

    def update_value(self, value):
        """Update value of the ScrollBar."""
        # Mark change as external to avoid setting YRange again in _channel_changed
        self.external_change = True
        self.setValue(value)
        self.external_change = False

    def update_nchan(self):
        """Update bar size."""
        if getattr(self.mne, "group_by", None) in ["position", "selection"]:
            self.setPageStep(1)
            self.setMaximum(len(self.mne.ch_selections) - 1)
        else:
            self.setPageStep(self.mne.n_channels)
            self.setMaximum(self.mne.ymax - self.mne.n_channels - 1)

    def keyPressEvent(self, event):
        """Customize key press events."""
        # Let main handle the keypress
        event.ignore()
