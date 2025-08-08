# License: BSD-3-Clause
# Copyright the MNE Qt Browser contributors.

import weakref
from collections import OrderedDict

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from mne import channel_indices_by_type
from mne.io.pick import _DATA_CH_TYPES_SPLIT
from mne.viz import plot_sensors
from mne.viz.utils import _figure_agg
from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtGui import QPainter, QPainterPath
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mne_qt_browser._utils import (
    _calc_chan_type_to_physical,
    _calc_data_unit_to_physical,
    _convert_physical_units,
    _disconnect,
    _get_channel_scaling,
    _methpartial,
    _q_font,
    _screen_geometry,
    _set_window_flags,
)


class _BaseDialog(QDialog):
    def __init__(
        self,
        main,
        widget=None,
        modal=False,
        name=None,
        title=None,
        flags=Qt.Window | Qt.Tool,
    ):
        super().__init__(main, flags)
        _set_window_flags(self)
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
        if hasattr(self, "name") and hasattr(self, "mne"):
            if self.name is not None and hasattr(self.mne, self.name):
                setattr(self.mne, self.name, None)
            if self in self.mne.child_figs:
                self.mne.child_figs.remove(self)
        event.accept()


class _ChannelFig(FigureCanvasQTAgg):
    def __init__(self, figure, mne):
        self.figure = figure
        self.mne = mne
        super().__init__(figure)
        _set_window_flags(self)
        self.setFocusPolicy(Qt.FocusPolicy(Qt.StrongFocus | Qt.WheelFocus))
        self.setFocus()
        self._lasso_path = None
        # Only update when mouse is pressed
        self.setMouseTracking(False)

    def paintEvent(self, event):
        super().paintEvent(event)
        # Lasso drawing doesn't seem to work with Matplotlib, thus it is replicated
        # in Qt.
        if self._lasso_path is not None:
            painter = QPainter(self)
            painter.setPen(self.mne.mkPen("red", width=2))
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


class SettingsDialog(_BaseDialog):
    """Shows additional settings."""

    def __init__(self, main, title="Settings", **kwargs):
        super().__init__(main, title=title, **kwargs)

        layout = QFormLayout()

        # Antialiasing
        self.antialiasing_box = QCheckBox()
        self.antialiasing_box.setToolTip("Enable/Disable antialiasing.\n")
        self.antialiasing_box.setChecked(self.mne.antialiasing)
        self.antialiasing_box.stateChanged.connect(
            _methpartial(self._toggle_antialiasing)
        )
        layout.addRow("antialiasing", self.antialiasing_box)

        # Downsampling
        self.downsampling_box = QSpinBox()
        self.downsampling_box.setToolTip(
            "Set an integer as the downsampling"
            ' factor or "Auto" to get the factor'
            " from the visible range.\n"
            " Setting the factor 1 means no "
            "downsampling.\n"
            " Default is 1."
        )
        self.downsampling_box.setMinimum(0)
        self.downsampling_box.setSpecialValueText("Auto")
        self.downsampling_box.setValue(
            0 if self.mne.downsampling == "auto" else self.mne.downsampling
        )
        self.downsampling_box.valueChanged.connect(
            _methpartial(self._value_changed, value_name="downsampling")
        )
        layout.addRow("downsampling", self.downsampling_box)

        # Downsampling method
        self.ds_method_cmbx = QComboBox()
        self.ds_method_cmbx.setToolTip(
            "<h2>Downsampling Method</h2>"
            "<ul>"
            "<li>Subsample:<br>"
            "Only take every n-th sample.</li>"
            "<li>Mean:<br>"
            "Take the mean of n samples.</li>"
            "<li>Peak:<br>"
            "Draws a saw wave from the minimum to the maximum from a "
            "collection of n samples.</li>"
            "</ul>"
            "<i>(Those methods are adapted from PyQtGraph)</i><br>"
            'Default is "peak".'
        )
        self.ds_method_cmbx.addItems(["subsample", "mean", "peak"])
        self.ds_method_cmbx.setCurrentText(self.mne.ds_method)
        self.ds_method_cmbx.currentTextChanged.connect(
            _methpartial(self._value_changed, value_name="ds_method")
        )
        layout.addRow("ds_method", self.ds_method_cmbx)

        # Scrolling sensitivity
        self.scroll_sensitivity_slider = QSlider(Qt.Horizontal)
        self.scroll_sensitivity_slider.setMinimum(10)
        self.scroll_sensitivity_slider.setMaximum(1000)
        self.scroll_sensitivity_slider.setToolTip(
            "Set the sensitivity of the scrolling in horizontal direction. "
            "Adjust this value if the scrolling for example with an horizontal "
            "mouse wheel is too fast or too slow. Default is 100."
        )
        self.scroll_sensitivity_slider.setValue(self.mne.scroll_sensitivity)
        self.scroll_sensitivity_slider.valueChanged.connect(
            _methpartial(self._value_changed, value_name="scroll_sensitivity")
        )
        # Set default
        layout.addRow("horizontal scroll sensitivity", self.scroll_sensitivity_slider)

        layout.addItem(QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Get all unique channel types
        ordered_types = self.mne.ch_types[self.mne.ch_order]
        unique_type_idxs = np.unique(ordered_types, return_index=True)[1]
        ch_types_ordered = [
            ordered_types[idx]
            for idx in sorted(unique_type_idxs)
            if ordered_types[idx] in self.mne.unit_scalings
        ]

        # Grid layout for channel spinboxes and settings
        ch_grid_layout = QGridLayout()

        # Create dropdown to choose units
        self.physical_units_cmbx = QComboBox()
        self.physical_units_cmbx.addItems(["/ mm", "/ cm", "/ inch"])
        self.physical_units_cmbx.currentTextChanged.connect(
            _methpartial(
                self._update_spinbox_values, ch_type="all", source="unit_change"
            )
        )
        current_units = self.physical_units_cmbx.currentText().split()[-1]

        # Add subgroup box to show channel type scalings
        ch_scroll_box = QGroupBox("Channel Configuration")
        ch_scroll_box.setStyleSheet("QGroupBox { font-size: 12pt; }")
        self.ch_scaling_spinboxes = {}
        self.ch_sensitivity_spinboxes = {}
        self.ch_label_widgets = {}

        ch_grid_layout.addWidget(QLabel("Channel Type"), 0, 0)
        ch_grid_layout.addWidget(QLabel("Scaling"), 0, 1)
        ch_grid_layout.addWidget(QLabel("Sensitivity"), 0, 2)
        grid_row = 1
        for ch_type in ch_types_ordered:
            self.ch_label_widgets[ch_type] = QLabel(
                f"{ch_type} ({self.mne.units[ch_type]})"
            )

            # Make scaling spinbox first
            ch_scale_spinbox = QDoubleSpinBox()
            ch_scale_spinbox.setMinimumWidth(100)
            ch_scale_spinbox.setRange(0, float("inf"))
            ch_scale_spinbox.setDecimals(1)
            ch_scale_spinbox.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
            ch_scale_spinbox.setValue(_get_channel_scaling(self, ch_type))
            ch_scale_spinbox.valueChanged.connect(
                _methpartial(
                    self._update_spinbox_values, ch_type=ch_type, source="scaling"
                )
            )
            self.ch_scaling_spinboxes[ch_type] = ch_scale_spinbox

            # Now make sensitivity spinbox
            ch_sens_spinbox = QDoubleSpinBox()
            ch_sens_spinbox.setMinimumWidth(100)
            ch_sens_spinbox.setRange(0, float("inf"))
            ch_sens_spinbox.setDecimals(1)
            ch_sens_spinbox.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
            ch_sens_spinbox.setReadOnly(False)
            ch_sens_spinbox.setDisabled(False)
            ch_sens_spinbox.setValue(
                _calc_chan_type_to_physical(self, ch_type, units=current_units)
            )
            ch_sens_spinbox.valueChanged.connect(
                _methpartial(
                    self._update_spinbox_values, ch_type=ch_type, source="sensitivity"
                )
            )
            self.ch_sensitivity_spinboxes[ch_type] = ch_sens_spinbox

            # Add these to the layout
            ch_grid_layout.addWidget(self.ch_label_widgets[ch_type], grid_row, 0)
            ch_grid_layout.addWidget(ch_scale_spinbox, grid_row, 1)
            ch_grid_layout.addWidget(ch_sens_spinbox, grid_row, 2)
            grid_row += 1

        ch_grid_layout.addWidget(self.physical_units_cmbx, grid_row, 2)
        ch_scroll_box.setLayout(ch_grid_layout)
        layout.addRow(ch_scroll_box)

        layout.addItem(QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Add box for monitor settings
        monitor_layout = QGridLayout()
        monitor_box = QGroupBox("Monitor Size")
        monitor_box.setStyleSheet("QGroupBox { font-size: 12pt; }")

        # Monitor height spinbox
        self.mon_height_spinbox = QDoubleSpinBox()
        self.mon_height_spinbox.setMinimumWidth(100)
        self.mon_height_spinbox.setRange(0, float("inf"))
        self.mon_height_spinbox.setDecimals(2)
        self.mon_height_spinbox.lineEdit().returnPressed.connect(
            _methpartial(self._update_monitor, dim="height")
        )
        monitor_layout.addWidget(QLabel("Monitor Height"), 0, 0)
        monitor_layout.addWidget(self.mon_height_spinbox, 0, 1)

        # Monitor width spinbox
        self.mon_width_spinbox = QDoubleSpinBox()
        self.mon_width_spinbox.setMinimumWidth(100)
        self.mon_width_spinbox.setRange(0, float("inf"))
        self.mon_width_spinbox.setDecimals(2)
        self.mon_width_spinbox.lineEdit().returnPressed.connect(
            _methpartial(self._update_monitor, dim="width")
        )
        monitor_layout.addWidget(QLabel("Monitor Width"), 1, 0)
        monitor_layout.addWidget(self.mon_width_spinbox, 1, 1)

        # DPI Spinbox
        self.dpi_spinbox = QDoubleSpinBox()
        self.dpi_spinbox.setMinimumWidth(100)
        self.dpi_spinbox.setRange(0, float("inf"))
        self.dpi_spinbox.setDecimals(2)
        self.dpi_spinbox.setReadOnly(True)
        self.dpi_spinbox.lineEdit().returnPressed.connect(
            _methpartial(self._update_monitor, dim="dpi")
        )
        monitor_layout.addWidget(QLabel("Monitor DPI"), 2, 0)
        monitor_layout.addWidget(self.dpi_spinbox, 2, 1)

        # Units combobox
        self.mon_units_cmbx = QComboBox()
        self.mon_units_cmbx.addItems(["mm", "cm", "inch"])
        self.current_monitor_units = self.mon_units_cmbx.currentText().split()[-1]
        self.mon_units_cmbx.currentTextChanged.connect(
            _methpartial(self._update_monitor, dim="unit_change")
        )
        monitor_layout.addWidget(QLabel("Monitor Units"), 3, 0)
        monitor_layout.addWidget(self.mon_units_cmbx, 3, 1)

        # Push buttons
        self.mon_reset_bttn = QPushButton("Reset")
        self.mon_reset_bttn.clicked.connect(self._reset_monitor_spinboxes)
        monitor_layout.addWidget(self.mon_reset_bttn, 4, 0, 1, 2)

        self._reset_monitor_spinboxes()
        monitor_box.setLayout(monitor_layout)
        layout.addRow(monitor_box)

        self.setLayout(layout)
        self.show()

    def closeEvent(self, event):  # noqa: D102
        _disconnect(self.ds_method_cmbx.currentTextChanged)
        _disconnect(self.scroll_sensitivity_slider.valueChanged)
        super().closeEvent(event)

    def _value_changed(self, new_value, value_name):
        if value_name == "downsampling" and new_value == 0:
            new_value = "auto"

        setattr(self.mne, value_name, new_value)

        if value_name == "scroll_sensitivity":
            self.mne.ax_hscroll._update_scroll_sensitivity()
        else:
            self.weakmain()._redraw()

    def _toggle_antialiasing(self, _):
        self.weakmain()._toggle_antialiasing()

    def _update_monitor(self, *args, dim="height"):
        dpr = QApplication.primaryScreen().devicePixelRatio()
        px_height = QApplication.primaryScreen().size().height()
        px_width = QApplication.primaryScreen().size().width()
        if dim == "height":
            new_ht_val = self.mon_height_spinbox.value()

            # Get new dpi
            mon_units = self.current_monitor_units
            mon_height_inch = _convert_physical_units(
                new_ht_val, from_unit=mon_units, to_unit="inch"
            )
            dpi = px_height / mon_height_inch  # / dpr

            # Find new width of monitor
            with QSignalBlocker(self.mon_width_spinbox):
                mon_width = self.mne.aspect_ratio * new_ht_val
                self.mon_width_spinbox.setValue(mon_width)

            self.mne.dpi = dpi
            self.dpi_spinbox.setValue(self.mne.dpi)

            self._update_spinbox_values(ch_type="all", source="unit_change")

        elif dim == "width":
            new_wd_value = self.mon_width_spinbox.value()

            # Get new dpi
            mon_units = self.current_monitor_units
            mon_width_inch = _convert_physical_units(
                new_wd_value, from_unit=mon_units, to_unit="inch"
            )
            dpi = px_width / mon_width_inch  # / dpr

            # Find new height of monitor
            with QSignalBlocker(self.mon_height_spinbox):
                mon_height = new_wd_value / self.mne.aspect_ratio
                self.mon_height_spinbox.setValue(mon_height)

            self.mne.dpi = dpi
            self.dpi_spinbox.setValue(self.mne.dpi)

            self._update_spinbox_values(ch_type="all", source="unit_change")

        elif dim == "unit_change":
            old_units = self.current_monitor_units
            new_units = self.mon_units_cmbx.currentText()

            mon_height_units = _convert_physical_units(
                self.mon_height_spinbox.value(), from_unit=old_units, to_unit=new_units
            )

            mon_width_units = _convert_physical_units(
                self.mon_width_spinbox.value(), from_unit=old_units, to_unit=new_units
            )

            with QSignalBlocker(self.mon_width_spinbox):
                self.mon_width_spinbox.setValue(mon_width_units)

            with QSignalBlocker(self.mon_height_spinbox):
                self.mon_height_spinbox.setValue(mon_height_units)

            self.current_monitor_units = new_units

        elif dim == "dpi":
            new_value = self.dpi_spinbox.value()
            self.mne.dpi = new_value
            mon_units = self.current_monitor_units

            with QSignalBlocker(self.mon_height_spinbox):
                mon_height_inch = (px_height / dpr) / new_value
                self.mon_height_spinbox.setValue(
                    _convert_physical_units(
                        mon_height_inch, from_unit="inch", to_unit=mon_units
                    )
                )

            with QSignalBlocker(self.mon_width_spinbox):
                mon_width_inch = (px_width / dpr) / new_value
                self.mon_width_spinbox.setValue(
                    _convert_physical_units(
                        mon_width_inch, from_unit="inch", to_unit=mon_units
                    )
                )

            self._update_spinbox_values(ch_type="all", source="unit_change")

        else:
            raise ValueError(f"Unknown dimension: {dim}")

    def _reset_monitor_spinboxes(self):
        """Reset monitor spinboxes to expected values."""
        mon_units = self.mon_units_cmbx.currentText()

        # Get the screen size
        height_mm = QApplication.primaryScreen().physicalSize().height()
        width_mm = QApplication.primaryScreen().physicalSize().width()

        height_mon_units = _convert_physical_units(
            height_mm, from_unit="mm", to_unit=mon_units
        )
        width_mon_units = _convert_physical_units(
            width_mm, from_unit="mm", to_unit=mon_units
        )

        self.mne.dpi = QApplication.primaryScreen().physicalDotsPerInch()

        # Set the spinbox values as such
        self.mon_height_spinbox.setValue(height_mon_units)
        self.mon_width_spinbox.setValue(width_mon_units)
        self.dpi_spinbox.setValue(self.mne.dpi)

        # Update sensitivity spinboxes
        self._update_spinbox_values(ch_type="all", source="unit_change")

    def _update_spinbox_values(self, *args, **kwargs):
        """Update spinbox values."""
        ch_type = kwargs["ch_type"]
        source = kwargs["source"]
        current_units = self.physical_units_cmbx.currentText().split()[-1]

        # A new value is passed in
        if len(args) > 0:
            new_value = args[0]

            # If source is scaling, update scaling and block signal to avoid recursion
            if source == "scaling":
                if new_value == 0:
                    self.mne.scalings[ch_type] = 1e-12
                else:
                    self.mne.scalings[ch_type] = 1
                    self.mne.scalings[ch_type] = new_value / _get_channel_scaling(
                        self, ch_type
                    )

                with QSignalBlocker(self.ch_sensitivity_spinboxes[ch_type]):
                    self.ch_sensitivity_spinboxes[ch_type].setValue(
                        _calc_chan_type_to_physical(self, ch_type, units=current_units)
                    )

                self.mne.scalebar_texts[ch_type].update_value()

            elif source == "sensitivity":
                # If new_value is 0 then scaling is stuck on 0; calculate what the new
                # scalings value will have to be
                if new_value == 0:
                    self.mne.scalings[ch_type] = 1e-12
                else:
                    scaler = 1 if self.mne.butterfly else 2
                    self.mne.scalings[ch_type] = (
                        new_value
                        * self.mne.scale_factor
                        * _calc_data_unit_to_physical(self, units=current_units)
                        / (scaler * self.mne.unit_scalings[ch_type])
                    )

                with QSignalBlocker(self.ch_scaling_spinboxes[ch_type]):
                    self.ch_scaling_spinboxes[ch_type].setValue(
                        _get_channel_scaling(self, ch_type)
                    )

                self.mne.scalebar_texts[ch_type].update_value()

            elif source == "unit_change":
                new_unit = new_value.split()[-1]
                ch_types = self.ch_scaling_spinboxes.keys()
                for ch_type in ch_types:
                    with QSignalBlocker(self.ch_sensitivity_spinboxes[ch_type]):
                        self.ch_sensitivity_spinboxes[ch_type].setValue(
                            _calc_chan_type_to_physical(self, ch_type, units=new_unit)
                        )

            else:
                raise ValueError(
                    f"Unknown source: {source}. "
                    f"Must be scaling or sensitivity. if specifying a new value"
                )

            self.mne.scalebar_texts[ch_type].update_value()
            self.weakmain()._redraw()
            # self.weakmain().scale_all(step=1, update_spinboxes=False)

        else:
            # Update all spinboxes
            ch_types = self.ch_scaling_spinboxes.keys()
            for ch_type in ch_types:
                with QSignalBlocker(self.ch_scaling_spinboxes[ch_type]):
                    self.ch_scaling_spinboxes[ch_type].setValue(
                        _get_channel_scaling(self, ch_type)
                    )
                with QSignalBlocker(self.ch_sensitivity_spinboxes[ch_type]):
                    self.ch_sensitivity_spinboxes[ch_type].setValue(
                        _calc_chan_type_to_physical(self, ch_type, units=current_units)
                    )


class HelpDialog(_BaseDialog):
    """Shows all keyboard-shortcuts."""

    def __init__(self, main, **kwargs):
        super().__init__(main, title="Help", **kwargs)

        # Show all keyboard shortcuts in a scroll area
        layout = QVBoxLayout()
        keyboard_label = QLabel("Keyboard Shortcuts")
        keyboard_label.setFont(_q_font(16, bold=True))
        layout.addWidget(keyboard_label)

        scroll_area = QScrollArea()
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )
        scroll_widget = QWidget()
        form_layout = QFormLayout()
        for key in main.mne.keyboard_shortcuts:
            key_dict = main.mne.keyboard_shortcuts[key]
            if "description" in key_dict:
                if "alias" in key_dict:
                    key = key_dict["alias"]
                for idx, key_des in enumerate(key_dict["description"]):
                    key_name = key
                    if "modifier" in key_dict:
                        mod = key_dict["modifier"][idx]
                        if mod is not None:
                            key_name = mod + " + " + key_name
                    form_layout.addRow(key_name, QLabel(key_des))
        scroll_widget.setLayout(form_layout)
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        # Additional help for mouse interaction
        inst = self.mne.instance_type
        is_raw = inst == "raw"
        is_epo = inst == "epochs"
        is_ica = inst == "ica"
        ch_cmp = "component" if is_ica else "channel"
        ch_epo = "epoch" if is_epo else "channel"
        ica_bad = "Mark/unmark component for exclusion"
        lclick_data = ica_bad if is_ica else f"Mark/unmark bad {ch_epo}"
        lclick_name = ica_bad if is_ica else "Mark/unmark bad channel"
        ldrag = "add annotation (in annotation mode)" if is_raw else None
        rclick_name = dict(
            ica="Show diagnostics for component",
            epochs="Show imageplot for channel",
            raw="Show channel location",
        )[inst]
        mouse_help = [
            (f"Left-click {ch_cmp} name", lclick_name),
            (f"Left-click {ch_cmp} data", lclick_data),
            ("Left-click-and-drag on plot", ldrag),
            ("Left-click on plot background", "Place vertical guide"),
            ("Right-click on plot background", "Clear vertical guide"),
            ("Right-click on channel name", rclick_name),
        ]

        mouse_label = QLabel("Mouse Interaction")
        mouse_label.setFont(_q_font(16, bold=True))
        layout.addWidget(mouse_label)
        mouse_widget = QWidget()
        mouse_layout = QFormLayout()
        for interaction, description in mouse_help:
            if description is not None:
                mouse_layout.addRow(f"{interaction}:", QLabel(description))
        mouse_widget.setLayout(mouse_layout)
        layout.addWidget(mouse_widget)

        self.setLayout(layout)
        self.show()

        # Set minimum width to avoid horizontal scrolling
        scroll_area.setMinimumWidth(
            scroll_widget.minimumSizeHint().width()
            + scroll_area.verticalScrollBar().width()
        )
        self.update()


class ProjDialog(_BaseDialog):
    """A dialog to toggle projections."""

    def __init__(self, main, *, name):
        self.external_change = True
        # Create projection layout
        super().__init__(main.window(), name=name, title="Projectors")

        layout = QVBoxLayout()
        labels = [p["desc"] for p in self.mne.projs]
        for ix, active in enumerate(self.mne.projs_active):
            if active:
                labels[ix] += " (already applied)"

        # make title
        layout.addWidget(
            QLabel(
                "Mark projectors applied on the plot.\n(Applied projectors are dimmed)."
            )
        )

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

        self.toggle_all_bt = QPushButton("Toggle All")
        self.toggle_all_bt.clicked.connect(self.toggle_all)
        layout.addWidget(self.toggle_all_bt)
        self.setLayout(layout)
        self.show()

    def _proj_changed(self, state=None, idx=None):
        # Only change if proj wasn't already applied
        if not self.mne.projs_active[idx]:
            self.mne.projs_on[idx] = not self.mne.projs_on[idx]
            self.weakmain()._apply_update_projectors()

    def toggle_all(self):
        """Toggle all projectors."""
        self.weakmain()._apply_update_projectors(toggle_all=True)

        # Update all checkboxes
        for idx, chkbx in enumerate(self.checkboxes):
            chkbx.setChecked(bool(self.mne.projs_on[idx]))


class SelectionDialog(_BaseDialog):  # noqa: D101
    def __init__(self, main):
        # Create widget
        super().__init__(main, name="fig_selection", title="Channel selection")
        geo = _screen_geometry(self)
        # Position selection dialog at right border of active screen
        xpos = geo.x() + geo.width() - 400
        self.setGeometry(xpos, 100, 400, 800)

        layout = QVBoxLayout()

        # Add channel plot
        fig = _figure_agg(figsize=(6, 6), dpi=96)
        ax = fig.add_axes([0, 0, 1, 1])
        self.channel_fig = plot_sensors(
            self.mne.info,
            kind="select",
            ch_type="all",
            title="",
            ch_groups=self.mne.group_by,
            axes=ax,
            show=False,
        )[0]
        self.channel_fig.lasso.callbacks.append(self._set_custom_selection)
        self.channel_widget = _ChannelFig(self.channel_fig, self.mne)
        layout.addWidget(self.channel_widget)

        selections_dict = self.mne.ch_selections
        selections_dict.update(Custom=np.array([], dtype=int))  # for lasso

        self.chkbxs = OrderedDict()
        for label in selections_dict:
            chkbx = QRadioButton(label)
            chkbx.clicked.connect(_methpartial(self._chkbx_changed, label=label))
            self.chkbxs[label] = chkbx
            layout.addWidget(chkbx)

        self.mne.old_selection = list(selections_dict)[0]
        self.chkbxs[self.mne.old_selection].setChecked(True)

        self._update_highlighted_sensors()

        # add instructions at bottom
        instructions = (
            "To use a custom selection, first click-drag on the sensor plot "
            'to "lasso" the sensors you want to select, or hold Ctrl while '
            "clicking individual sensors. Holding Ctrl while click-dragging "
            "allows a lasso selection adding to (rather than replacing) the "
            "existing selection."
        )
        help_widget = QTextEdit(instructions)
        help_widget.setReadOnly(True)
        layout.addWidget(help_widget)

        self.setLayout(layout)
        self.show(center=False)

    def _chkbx_changed(self, checked=True, label=None):
        # _chkbx_changed is called either directly (with checked=None) or through
        # _methpartial with a Qt signal. The signal includes the bool argument
        # 'checked'. Old versions of MNE-Python tests will call this function directly
        # without the checked argument _chkbx_changed(label), thus it has to be wrapped
        # in case only one argument is provided to retain compatibility of the tests
        # between new/old versions of mne-qt-browser and MNE-Python.
        if label is None:
            label = checked
        # Disable butterfly if checkbox is clicked
        if self.mne.butterfly:
            self.weakmain()._set_butterfly(False)
        if label == "Custom" and not len(self.mne.ch_selections["Custom"]):
            label = self.mne.old_selection
        # Select the checkbox no matter if clicked on when active or not
        self.chkbxs[label].setChecked(True)
        # Update selections
        self.mne.old_selection = label
        self.mne.picks = np.asarray(self.mne.ch_selections[label])
        self.mne.n_channels = len(self.mne.picks)
        # Update highlighted sensors
        self._update_highlighted_sensors()
        # if "Vertex" is defined, some channels appear twice, so if "Vertex" is
        # selected, ch_start should be the *first* match; otherwise it should be the
        # *last* match (since "Vertex" is always the first selection group, if it
        # exists).
        if label == "Custom":
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
        self.mne.plt.setYRange(
            self.mne.ch_start, self.mne.ch_start + self.mne.n_channels + 1, padding=0
        )

        # Update scrollbar
        label_idx = list(self.mne.ch_selections).index(label)
        self.mne.ax_vscroll.update_value(label_idx)

        # Update all y-positions, because channels can appear in multiple selections on
        # different y-positions
        for trace in self.mne.traces:
            trace.update_ypos()
            trace.update_data()

    def _set_custom_selection(self):
        chs = self.channel_fig.lasso.selection
        inds = np.isin(self.mne.ch_names, chs)
        self.mne.ch_selections["Custom"] = inds.nonzero()[0]
        if any(inds):
            self._chkbx_changed(None, "Custom")

    def _update_highlighted_sensors(self):
        inds = np.isin(
            self.mne.fig_selection.channel_fig.lasso.ch_names,
            self.mne.ch_names[self.mne.picks],
        ).nonzero()[0]
        self.channel_fig.lasso.select_many(inds)
        self.channel_widget.draw()

    def _update_bad_sensors(self, pick, mark_bad):
        sensor_picks = list()
        ch_indices = channel_indices_by_type(self.mne.info)
        for this_type in _DATA_CH_TYPES_SPLIT:
            if this_type in self.mne.ch_types:
                sensor_picks.extend(ch_indices[this_type])
        sensor_idx = np.isin(sensor_picks, pick).nonzero()[0]
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
        name_idx = list(self.mne.ch_selections).index(self.mne.old_selection)
        new_idx = np.clip(name_idx + step, 0, len(self.mne.ch_selections) - 1)
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
        if hasattr(self.channel_fig.lasso, "callbacks"):
            self.channel_fig.lasso.callbacks.clear()
        for chkbx in self.chkbxs.values():
            _disconnect(chkbx.clicked, allow_error=True)
        main = self.weakmain()
        if main is not None:
            main.close()


class _AnnotEditDialog(_BaseDialog):
    def __init__(self, annot_dock):
        super().__init__(annot_dock.weakmain(), title="Edit Annotations")
        self.ad = annot_dock

        self.current_mode = None

        layout = QVBoxLayout()
        self.descr_label = QLabel()
        if self.mne.selected_region:
            self.mode_cmbx = QComboBox()
            self.mode_cmbx.addItems(["all", "selected"])
            self.mode_cmbx.currentTextChanged.connect(self._mode_changed)
            layout.addWidget(QLabel("Edit Scope:"))
            layout.addWidget(self.mode_cmbx)
        # Set group as default
        self._mode_changed("all")

        layout.addWidget(self.descr_label)
        self.input_w = QLineEdit()
        layout.addWidget(self.input_w)
        bt_layout = QHBoxLayout()
        ok_bt = QPushButton("Ok")
        ok_bt.clicked.connect(self._edit)
        bt_layout.addWidget(ok_bt)
        cancel_bt = QPushButton("Cancel")
        cancel_bt.clicked.connect(self.close)
        bt_layout.addWidget(cancel_bt)
        layout.addLayout(bt_layout)
        self.setLayout(layout)
        self.show()

    def _mode_changed(self, mode):
        self.current_mode = mode
        if mode == "all":
            curr_des = self.ad.description_cmbx.currentText()
        else:
            curr_des = self.mne.selected_region.description
        self.descr_label.setText(f'Change "{curr_des}" to:')

    def _edit(self):
        new_des = self.input_w.text()
        if new_des:
            if self.current_mode == "all" or self.mne.selected_region is None:
                self.ad._edit_description_all(new_des)
            else:
                self.ad._edit_description_selected(new_des)
            self.close()
