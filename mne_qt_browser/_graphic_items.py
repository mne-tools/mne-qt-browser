import functools
import weakref

import numpy as np
from matplotlib.colors import to_rgba_array
from mne.utils import logger, warn
from mne.viz.utils import _simplify_float
from pyqtgraph import (
    FillBetweenItem,
    InfiniteLine,
    InfLineLabel,
    LinearRegionItem,
    PlotCurveItem,
    PlotDataItem,
    Point,
    TextItem,
    mkBrush,
)
from qtpy.QtCore import QLineF, QSignalBlocker, Qt, Signal
from qtpy.QtGui import QTransform
from qtpy.QtWidgets import QGraphicsLineItem

from mne_qt_browser._colors import _get_color
from mne_qt_browser._utils import _get_channel_scaling, _methpartial, _q_font

_vline_color = (0, 191, 0)


def propagate_to_children(method):  # noqa: D103
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        propagate = kwargs.pop("propagate", True)
        result = method(*args, **kwargs)
        if args[0].mne.is_epochs and propagate:
            # parent always goes first
            if hasattr(args[0], "child_traces"):
                for child_trace in args[0].child_traces:
                    getattr(child_trace, method.__name__)(*args[1:], **kwargs)
        return result

    return wrapper


class AnnotRegion(LinearRegionItem):
    """Graphics object for annotations."""

    regionChangeFinished = Signal(object)
    gotSelected = Signal(object)
    removeRequested = Signal(object)
    removeSingleChannelAnnots = Signal(object)
    sigToggleVisibility = Signal(bool)
    sigUpdateColor = Signal(object)  # can be str or tuple

    def __init__(self, mne, description, values, weakmain, ch_names=None):
        super().__init__(
            values=values,
            orientation="vertical",
            movable=True,
            swapMode="sort",
            bounds=(0, mne.xmax + 1 / mne.info["sfreq"]),
        )
        # Set default z-value to 0 to be behind other items in scene
        self.setZValue(0)

        self.sigRegionChangeFinished.connect(self._region_changed)
        self.weakmain = weakmain
        self.mne = mne
        self.description = description
        self.old_onset = values[0]
        self.selected = False

        self.label_item = TextItem(text=description, anchor=(0.5, 0.5))
        self.label_item.setFont(_q_font(10, bold=True))
        self.sigRegionChanged.connect(self.update_label_pos)

        self.update_color(all_channels=(not ch_names))

        self.single_channel_annots = {}
        if ch_names is not None and len(ch_names):
            for ch in ch_names:
                self._add_single_channel_annot(ch)

        self.mne.plt.addItem(self, ignoreBounds=True)
        self.mne.plt.addItem(self.label_item, ignoreBounds=True)

    def _region_changed(self):
        # Check for overlapping regions
        overlap_has_sca = []
        overlapping_regions = list()
        for region in self.mne.regions:
            if region.description != self.description or id(self) == id(region):
                continue
            values = region.getRegion()
            if (
                any(self.getRegion()[0] <= val <= self.getRegion()[1] for val in values)
                or (values[0] <= self.getRegion()[0] <= values[1])
                and (values[0] <= self.getRegion()[1] <= values[1])
            ):
                overlapping_regions.append(region)
                overlap_has_sca.append(len(region.single_channel_annots) > 0)

        # Terminate if this or an overlapping region has channel-specific annotations
        if (len(self.single_channel_annots) > 0 or any(overlap_has_sca)) and len(
            overlapping_regions
        ) > 0:
            dur = self.getRegion()[1] - self.getRegion()[0]
            self.setRegion((self.old_onset, self.old_onset + dur))
            warn("Can not combine channel-based annotations with any other annotation.")
            return

        # figure out new boundaries
        regions_ = np.array(
            [region.getRegion() for region in overlapping_regions] + [self.getRegion()]
        )

        self.regionChangeFinished.emit(self)

        onset = np.min(regions_[:, 0])
        offset = np.max(regions_[:, 1])

        self.old_onset = onset

        logger.debug(f"New {self.description} region: {onset:.2f} - {offset:.2f}")
        # remove overlapping regions
        for region in overlapping_regions:
            self.weakmain()._remove_region(region, from_annot=False)
        # re-set while blocking the signal to avoid re-running this function
        with QSignalBlocker(self):
            self.setRegion((onset, offset))

        self.update_label_pos()

    def _add_single_channel_annot(self, ch_name):
        self.single_channel_annots[ch_name] = SingleChannelAnnot(
            self.mne, self.weakmain, self, ch_name
        )

    def _remove_single_channel_annot(self, ch_name):
        self.single_channel_annots[ch_name].remove()
        self.single_channel_annots.pop(ch_name)

    def _toggle_single_channel_annot(self, ch_name, update_color=True):
        """Add or remove single channel annotations."""
        # Exit if MNE-Python not updated to support shift-click
        if not hasattr(self.weakmain(), "_toggle_single_channel_annotation"):
            warn(
                "MNE must be updated to version 1.8 or above to "
                "support add/remove channels from annotation."
            )
            return

        region_idx = self.weakmain()._get_onset_idx(self.getRegion()[0])
        self.weakmain()._toggle_single_channel_annotation(ch_name, region_idx)
        if ch_name not in self.single_channel_annots.keys():
            self._add_single_channel_annot(ch_name)
        else:
            self._remove_single_channel_annot(ch_name)

        if update_color:
            self.update_color(
                all_channels=(not list(self.single_channel_annots.keys()))
            )

    def update_color(self, all_channels=True):
        """Update color of annotation region.

        Parameters
        ----------
        all_channels : bool
            all_channels should be False for channel specific annotations. These
            annotations will be more transparent with a dashed outline.
        """
        color_string = self.mne.annotation_segment_colors[self.description]
        self.base_color = _get_color(color_string, self.mne.dark)
        self.hover_color = _get_color(color_string, self.mne.dark)
        self.text_color = _get_color(color_string, self.mne.dark)
        self.base_color.setAlpha(75 if all_channels else 15)
        self.hover_color.setAlpha(150)
        self.text_color.setAlpha(255)
        kwargs = dict(color=self.hover_color, width=2)
        if not all_channels:
            color = _get_color(color_string, self.mne.dark)
            color.setAlpha(75)
            kwargs.update(
                style=Qt.CustomDashLine,
                cap=Qt.FlatCap,
                dash=[8, 8],
                color=color,
            )
        self.line_pen = self.mne.mkPen(**kwargs)
        self.hover_pen = self.mne.mkPen(color=self.text_color, width=2)
        self.setBrush(self.base_color)
        self.setHoverBrush(self.hover_color)
        self.label_item.setColor(self.text_color)
        for line in self.lines:
            line.setPen(self.line_pen)
            line.setHoverPen(self.hover_pen)
        self.update()
        self.sigUpdateColor.emit(color_string)

    def update_description(self, description):
        """Update description of annotation region."""
        self.description = description
        self.label_item.setText(description)
        self.label_item.update()

    def update_visible(self, visible):
        """Update if annotation region is visible."""
        self.setVisible(visible)
        self.label_item.setVisible(visible)
        self.sigToggleVisibility.emit(visible)

    def remove(self):
        """Remove annotation region."""
        self.removeSingleChannelAnnots.emit(self)
        self.removeRequested.emit(self)
        vb = self.mne.viewbox
        if vb and self.label_item in vb.addedItems:
            vb.removeItem(self.label_item)

    def select(self, selected):
        """Update select state of annotation region."""
        self.selected = selected
        if selected:
            self.label_item.setColor("w")
            self.label_item.fill = mkBrush(self.hover_color)
            self.gotSelected.emit(self)
        else:
            self.label_item.setColor(self.text_color)
            self.label_item.fill = mkBrush(None)
        logger.debug(
            f"{'Selected' if self.selected else 'Deselected'} annotation: "
            f"{self.description}"
        )
        self.label_item.update()

    def mouseClickEvent(self, event):
        """Customize mouse click events."""
        if self.mne.annotation_mode and (
            event.button() == Qt.LeftButton and event.modifiers() & Qt.ShiftModifier
        ):
            scene_pos = self.mapToScene(event.pos())

            for t in self.mne.traces:
                trace_path = t.shape()
                trace_point = t.mapFromScene(scene_pos)
                if trace_path.contains(trace_point):
                    self._toggle_single_channel_annot(t.ch_name)
                    event.accept()
                    break

        elif self.mne.annotation_mode:
            if event.button() == Qt.LeftButton and self.movable:
                logger.debug(f"Mouse event in annotation mode for {event.pos()}...")
                self.select(True)
                event.accept()
            elif event.button() == Qt.RightButton and self.movable:
                self.remove()
                # the annotation removed should be the one on top of all others, which
                # should correspond to the one of the type currently selected and with
                # the highest z-value
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

        with QSignalBlocker(self.lines[0]):
            for pos, line in zip(new_pos, self.lines):
                line.setPos(pos)
        self.prepareGeometryChange()

        if ev.isFinish():
            self.moving = False
            self.sigRegionChangeFinished.emit(self)
        else:
            self.sigRegionChanged.emit(self)

    def update_label_pos(self):
        """Update position of description label from annotation region."""
        rgn = self.getRegion()
        vb = self.mne.viewbox
        if vb:
            ymax = vb.viewRange()[1][1]
            self.label_item.setPos(sum(rgn) / 2, ymax - 0.3)


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
            ch_type_idxs = np.where(self.mne.ch_types[self.mne.picks] == self.ch_type)[
                0
            ]

            for idx in ch_type_idxs:
                ch_name = self.mne.ch_names[self.mne.picks[idx]]
                if (
                    ch_name not in self.mne.info["bads"]
                    and ch_name not in self.mne.whitened_ch_names
                ):
                    self.ypos = self.mne.ch_start + idx + 1
                    break
            # Consider all indices bad
            if self.ypos is None:
                self.ypos = self.mne.ch_start + ch_type_idxs[0] + 1

    def update_x_position(self):
        """Update x-position of scalebar."""
        if self._is_visible():
            if self.ypos is None:
                self._get_ypos()
            self._set_position(self.mne.t_start, self.ypos)

    def update_y_position(self):
        """Update y-position of scalebar."""
        if self._is_visible():
            self.setVisible(True)
            self._get_ypos()
            self._set_position(self.mne.t_start, self.ypos)
        else:
            self.setVisible(False)


class Crosshair(InfiniteLine):
    """Continuously updating marker inside the trace plot."""

    def __init__(self, mne):
        super().__init__(angle=90, movable=False, pen="g")
        self.mne = mne
        self.y = 1

    def set_data(self, x, y):
        """Set x and y data for crosshair point."""
        self.setPos(x)
        self.y = y

    def paint(self, p, *args):  # noqa: D102
        super().paint(p, *args)

        p.setPen(self.mne.mkPen("r", width=4))
        p.drawPoint(Point(self.y, 0))


class DataTrace(PlotCurveItem):
    """Graphics object for single data trace."""

    def __init__(self, main, ch_idx, child_idx=None, parent_trace=None):
        super().__init__()
        self.weakmain = weakref.ref(main)
        self.mne = main.mne
        del main

        # Set clickable with small area around trace to make clicking easier
        self.setClickable(True, 12)

        # Set default z-value to 1 to be before other items in scene
        self.setZValue(1)

        # General attributes
        # The ch_idx is the index of the channel represented by this trace in the
        # channel-order from the unchanged instance (which also picks refer to)
        self.ch_idx = None
        # The range_idx is the index of the channel represented by this trace in the
        # shown range
        self.range_idx = None
        # The order_idx is the index of the channel represented by this trace in the
        # channel-order (defined e.g. by group_by)
        self.order_idx = None
        # Name of the channel the trace represents
        self.ch_name = None
        # Indicates if trace is bad
        self.isbad = None
        # Channel type of trace
        self.ch_type = None
        # Color specifier (all possible Matplotlib color formats)
        self.color = None

        # Attributes for epochs mode

        self.child_idx = child_idx  # Index of child if child
        self.parent_trace = parent_trace  # Reference to parent if child

        # Only for parent traces
        if self.parent_trace is None:
            self.mne.traces.append(self)
            self.child_traces = list()  # References to children
            self.trace_colors = None  # Colors of trace in viewrange

        # set attributes
        self.set_ch_idx(ch_idx)
        self.update_color()
        self.update_scale()
        # Avoid calling self.update_data() twice on initialization because of
        # update_scale()
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
            # (PlotCurveItem only supports one color per object). There are always as
            # many color-specific traces added depending on the whole time range of the
            # instance regardless of the currently visible time range (to avoid checking
            # for new colors while scrolling horizontally).

            # Only for parent trace
            if hasattr(self, "child_traces"):
                self.trace_colors = np.unique(
                    self.mne.epoch_color_ref[self.ch_idx], axis=0
                )
                n_childs = len(self.child_traces)
                trace_diff = len(self.trace_colors) - n_childs - 1
                # Add child traces if necessary
                if trace_diff > 0:
                    for cix in range(n_childs, n_childs + trace_diff):
                        child = DataTrace(
                            self.weakmain(),
                            self.ch_idx,
                            child_idx=cix,
                            parent_trace=self,
                        )
                        self.child_traces.append(child)
                elif trace_diff < 0:
                    for _ in range(abs(trace_diff)):
                        rm_trace = self.child_traces.pop()
                        rm_trace.remove()

                # Set parent color
                self.color = self.trace_colors[0]

            # Only for child trace
            else:
                self.color = self.parent_trace.trace_colors[self.child_idx + 1]

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
        """Update when view range or ch_idx changes."""
        self.range_idx = np.argwhere(self.mne.picks == self.ch_idx)[0][0]

    @propagate_to_children
    def update_ypos(self):  # noqa: D401
        """Update when butterfly is toggled or ch_idx changes."""
        if self.mne.butterfly and self.mne.fig_selection is not None:
            self.ypos = self.mne.selection_ypos_dict[self.ch_idx]
        elif self.mne.fig_selection is not None and self.mne.old_selection == "Custom":
            self.ypos = self.range_idx + 1
        elif self.mne.butterfly:
            self.ypos = self.mne.butterfly_type_order.index(self.ch_type) + 1
        else:
            self.ypos = self.range_idx + self.mne.ch_start + 1

    @propagate_to_children
    def update_scale(self):  # noqa: D102
        transform = QTransform()
        transform.scale(1.0, self.mne.scale_factor)
        self.setTransform(transform)

        if self.mne.clipping is not None:
            self.update_data(propagate=False)

    @propagate_to_children
    def set_ch_idx(self, ch_idx):
        """Set the channel index and all deriving indices."""
        # The ch_idx is the index of the channel represented by this trace in the
        # channel order from the unchanged instance (which also picks refer to)
        self.ch_idx = ch_idx
        # The range_idx is the index of the channel represented by this trace in the
        # shown range
        self.update_range_idx(propagate=False)
        # The order_idx is the index of the channel represented by this trace in the
        # channel order (defined e.g. by group_by).
        self.order_idx = np.argwhere(self.mne.ch_order == self.ch_idx)[0][0]
        self.ch_name = self.mne.inst.ch_names[ch_idx]
        self.isbad = self.ch_name in self.mne.info["bads"]
        self.ch_type = self.mne.ch_types[ch_idx]
        self.update_ypos(propagate=False)

    @propagate_to_children
    def update_data(self):
        """Update data (fetch data from self.mne according to self.ch_idx)."""
        if self.mne.is_epochs or (
            self.mne.clipping is not None and self.mne.clipping != "clamp"
        ):
            connect = "finite"
            skip = False
        else:
            connect = "all"
            skip = True

        if self.mne.data_precomputed:
            data = self.mne.data[self.order_idx]
            data /= self.mne.scalings[self.ch_type]
        else:
            data = self.mne.data[self.range_idx]
        times = self.mne.times

        # Get decim-specific time if enabled
        if self.mne.decim != 1:
            times = times[:: self.mne.decim_data[self.range_idx]]
            data = data[..., :: self.mne.decim_data[self.range_idx]]

        # For multiple color traces with epochs, replace other colors with NaN
        if self.mne.is_epochs:
            data = np.copy(data)
            check_color = self.mne.epoch_color_ref[self.ch_idx, self.mne.epoch_idx]
            bool_ixs = np.invert(np.equal(self.color, check_color).all(axis=1))
            starts = self.mne.boundary_times[self.mne.epoch_idx][bool_ixs]
            stops = self.mne.boundary_times[self.mne.epoch_idx + 1][bool_ixs]

            for start, stop in zip(starts, stops):
                data[np.logical_and(start <= times, times <= stop)] = np.nan

        assert times.shape[-1] == data.shape[-1]

        self.setData(
            times,
            data,
            connect=connect,
            skipFiniteCheck=skip,
            antialias=self.mne.antialiasing,
        )

        self.setPos(0, self.ypos)

    def toggle_bad(self, x=None):
        """Toggle bad status."""
        # Toggle bad epoch
        if self.mne.is_epochs and x is not None:
            epoch_idx, color = self.weakmain()._toggle_bad_epoch(x)

            # Update epoch color
            if color != "none":
                new_epo_color = np.repeat(
                    to_rgba_array(color), len(self.mne.inst.ch_names), axis=0
                )
            elif self.mne.epoch_colors is None:
                new_epo_color = np.concatenate(
                    [to_rgba_array(c) for c in self.mne.ch_color_ref.values()]
                )
            else:
                new_epo_color = np.concatenate(
                    [to_rgba_array(c) for c in self.mne.epoch_colors[epoch_idx]]
                )

            # Update bad channel colors
            bad_idxs = np.isin(self.mne.ch_names, self.mne.info["bads"])
            new_epo_color[bad_idxs] = to_rgba_array(self.mne.ch_color_bad)

            self.mne.epoch_color_ref[:, epoch_idx] = new_epo_color

            # Update overview bar
            self.mne.overview_bar.update_bad_epochs()

            # Update other traces inlcuding self
            for trace in self.mne.traces:
                trace.update_color()
                # Update data is necessary because colored segments will vary
                trace.update_data()

        # Toggle bad channel
        else:
            bad_color, pick, marked_bad = self.weakmain()._toggle_bad_channel(
                self.range_idx
            )
            self.weakmain()._apply_update_projectors()

            # Update line color status
            self.isbad = not self.isbad

            # Update colors for epochs
            if self.mne.is_epochs:
                if marked_bad:
                    new_ch_color = np.repeat(
                        to_rgba_array(bad_color), len(self.mne.inst), axis=0
                    )
                elif self.mne.epoch_colors is None:
                    ch_color = self.mne.ch_color_ref[self.ch_name]
                    new_ch_color = np.repeat(
                        to_rgba_array(ch_color), len(self.mne.inst), axis=0
                    )
                else:
                    new_ch_color = np.concatenate(
                        [to_rgba_array(c[pick]) for c in self.mne.epoch_colors]
                    )

                self.mne.epoch_color_ref[pick, :] = new_ch_color

            # Update trace color
            self.update_color()
            if self.mne.is_epochs:
                self.update_data()

            # Update channel axis
            self.weakmain()._update_yaxis_labels()

            # Update overview bar
            self.mne.overview_bar.update_bad_channels()

            # Update sensor color (if in selection mode)
            if self.mne.fig_selection is not None:
                self.mne.fig_selection._update_bad_sensors(pick, marked_bad)

    def mouseClickEvent(self, ev):
        """Customize mouse click events."""
        if (
            not self.clickable
            or ev.button() != Qt.MouseButton.LeftButton
            or self.mne.annotation_mode
        ):
            # Explicitly ignore events in annotation mode
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


class EventLine(InfiniteLine):
    """Displays Events inside Trace-Plot."""

    def __init__(self, mne, pos, label, color):
        super().__init__(
            pos,
            pen=color,
            movable=False,
            label=str(label),
            labelOpts={
                "position": 0.98,
                "color": color,
                "anchors": [(0, 0.5), (0, 0.5)],
            },
        )
        self.mne = mne
        self.label.setFont(_q_font(10, bold=True))
        self.setZValue(0)

        self.mne.plt.addItem(self)


class ScaleBar(BaseScaleBar, QGraphicsLineItem):  # noqa: D101
    def __init__(self, mne, ch_type):
        BaseScaleBar.__init__(self, mne, ch_type)
        QGraphicsLineItem.__init__(self)

        self.setZValue(1)
        pen = self.mne.mkPen(color="#AA3377", width=5)
        pen.setCapStyle(Qt.FlatCap)
        self.setPen(pen)
        self.update_y_position()

    def _set_position(self, x, y):
        self.setLine(QLineF(x, y - 0.5, x, y + 0.5))

    def get_ydata(self):
        """Get y-data for tests."""
        line = self.line()
        return line.y1(), line.y2()


class ScaleBarText(BaseScaleBar, TextItem):  # noqa: D101
    def __init__(self, mne, ch_type):
        BaseScaleBar.__init__(self, mne, ch_type)
        TextItem.__init__(self, color="#AA3377")

        self.setFont(_q_font(10))
        self.setZValue(2)  # To draw over RawTraceItems

        self.update_value()
        self.update_y_position()

    def update_value(self):
        """Update value of ScaleBarText."""
        inv_norm = _get_channel_scaling(self, self.ch_type)
        self.setText(f"{_simplify_float(inv_norm)} {self.mne.units[self.ch_type]}")

    def _set_position(self, x, y):
        self.setPos(x, y)


class SingleChannelAnnot(FillBetweenItem):
    def __init__(self, mne, weakmain, annot, ch_name):
        self.weakmain = weakmain
        self.mne = mne
        self.annot = annot
        self.ch_name = ch_name

        # We have a choice here: some channel-specific annotations will have
        # channels that are not plotted (e.g., with raw.plot(..., order=[0, 1])).
        # Here we choose to add them as invisible plot items so that if you for example
        # remove all *visible* channels from the annot it won't suddenly become an
        # all-channel annotation. It will instead disappear from the plot, but still
        # live in raw.annotations. We could emit a warning, but it would probably
        # become annoying. At some point we could add a warning
        # *when you delete the last visible channel* to let people know that the
        # annot still exists but can no longer be modified, or something similar.
        # It would be good to have this be driven by an actual use case / experience.
        idx = np.where(self.mne.ch_names[self.mne.ch_order] == self.ch_name)[0]
        if len(idx) == 1:  # should be 1 or 0 (if channel not plotted at all)
            self.ypos = idx + np.array([0.5, 1.5])
        else:
            self.ypos = np.full(2, np.nan)

        self.upper = PlotDataItem()
        self.lower = PlotDataItem()

        # init
        super().__init__(self.lower, self.upper)

        self.update_plot_curves()

        color_string = self.mne.annotation_segment_colors[self.annot.description]
        self.update_color(color_string)

        self.mne.plt.addItem(self, ignoreBounds=True)

        self.annot.removeSingleChannelAnnots.connect(self.remove)
        self.annot.sigRegionChangeFinished.connect(self.update_plot_curves)
        self.annot.sigRegionChanged.connect(self.update_plot_curves)
        self.annot.sigToggleVisibility.connect(self.update_visible)
        self.annot.sigUpdateColor.connect(self.update_color)

    def update_plot_curves(self):
        """Update the lower and upper bounds of the region."""
        # When using PlotCurveItem
        # annot_range = np.array(self.annot.getRegion())
        # self.lower.setData(x=annot_range, y=self.ypos[[0, 0]])
        # self.upper.setData(x=annot_range, y=self.ypos[[1, 1]])

        # When using PlotDataItem
        x_min, x_max = self.annot.getRegion()
        y_min, y_max = self.ypos
        self.upper.setData(x=(x_min, x_max), y=(y_max, y_max))
        self.lower.setData(x=(x_min, x_max), y=(y_min, y_min))

    def update_visible(self, visible):
        """Update visibility to match the annot."""
        self.setVisible(visible)

    def update_color(self, color_string=None):
        brush = _get_color(color_string, self.mne.dark)
        brush.setAlpha(60)
        self.setBrush(brush)

    def remove(self):
        """Remove this from plot."""
        vb = self.mne.viewbox
        vb.removeItem(self)


class VLine(InfiniteLine):
    """Marker to be placed inside the trace plot."""

    def __init__(self, mne, pos, bounds):
        super().__init__(
            pos,
            pen={"color": _vline_color, "width": 2},
            hoverPen="y",
            movable=True,
            bounds=bounds,
        )
        self.mne = mne
        self.label = VLineLabel(self)

    def setMouseHover(self, hover):
        """Customize the mouse hovering event."""
        super().setMouseHover(hover)
        # Also change color of label
        self.label.fill = self.currentPen.color()
        self.label.border = self.currentPen
        self.label.update()


class VLineLabel(InfLineLabel):
    """Label of the vline displaying the time."""

    def __init__(self, vline):
        super().__init__(
            vline,
            text="{value:.3f} s",
            position=0.98,
            fill=_vline_color,
            color="k",
            movable=True,
        )
        self.cursorOffset = None

    def mouseDragEvent(self, ev):
        """Customize mouse drag events."""
        if self.movable and ev.button() == Qt.LeftButton:
            if ev.isStart():
                self.line.moving = True
                self.cursorOffset = self.line.pos() - self.mapToView(ev.buttonDownPos())
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
            # Show epoch time
            t_vals_abs = np.linspace(
                0, self.line.mne.epoch_dur, len(self.line.mne.inst.times)
            )
            search_val = value % self.line.mne.epoch_dur
            t_idx = np.searchsorted(t_vals_abs, search_val)
            value = self.line.mne.inst.times[t_idx]
        self.setText(self.format.format(value=value))
        self.updatePosition()

    def hoverEvent(self, ev):
        _methpartial(self.line.hoverEvent)(ev)
