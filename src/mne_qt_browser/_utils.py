# License: BSD-3-Clause
# Copyright the MNE Qt Browser contributors.

import functools
import os
import warnings
import weakref

from mne.io.pick import _DATA_CH_TYPES_ORDER_DEFAULT
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QFont, QGuiApplication

# MNE's butterfly plots traditionally default to the channel ordering of mag, grad, ...,
# which is inconsistent with the order in non-butterfly mode and hence doesn't match the
# order in the overview bar either. So we swap grads and mags here.
DATA_CH_TYPES_ORDER = ("grad", "mag", *_DATA_CH_TYPES_ORDER_DEFAULT[2:])

qsettings_params = {
    "antialiasing": False,
    "scroll_sensitivity": 100,  # steps per view (relative to time)
    "downsampling": 1,
    "ds_method": "peak",
    "overview_mode": "channels",
    "overview_visible": True,
}

_unit_per_inch = dict(mm=25.4, cm=2.54, inch=1.0)


def _disconnect(sig, *, allow_error=False):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "Failed to disconnect", category=RuntimeWarning
            )
            sig.disconnect()
    except (TypeError, RuntimeError):  # if there are no connections, ignore it
        if not allow_error:
            raise


def _methpartial(meth, **kwargs):
    """Use WeakMethod to create a partial method."""
    meth = weakref.WeakMethod(meth)

    def call(*args_, **kwargs_):
        meth_ = meth()
        if meth_ is not None:
            return meth_(*args_, **kwargs, **kwargs_)

    return call


def _q_font(point_size, bold=False):
    font = QFont()
    font.setPointSize(point_size)
    font.setBold(bold)
    return font


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


def _screen_geometry(widget):
    try:
        # Qt 5.14+
        return widget.screen().geometry()
    except AttributeError:
        # Top center of the widget
        screen = QGuiApplication.screenAt(
            widget.mapToGlobal(QPoint(widget.width() // 2, 0))
        )
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        geometry = screen.geometry()

        return geometry


def _set_window_flags(widget):
    if os.getenv("_MNE_BROWSER_BACK", "").lower() == "true":
        widget.setWindowFlags(widget.windowFlags() | Qt.WindowStaysOnBottomHint)


def _calc_chan_type_to_physical(widget, ch_type, units="mm"):
    """Convert data to physical units."""
    return _get_channel_scaling(widget, ch_type) / _calc_data_unit_to_physical(
        widget, units=units
    )


def _calc_data_unit_to_physical(widget, units="mm"):
    """Calculate the physical size of a data unit."""
    # Get the ViewBox and its height in pixels
    vb = widget.mne.viewbox
    height_px = vb.geometry().height()

    # Get the view range in data units (here we write V for simplicity and dimensional
    # analysis but it works for any underlying data unit)
    view_range = vb.viewRange()
    height_V = view_range[1][1] - view_range[1][0]

    # Calculate the pixel-to-data ratio
    if height_V == 0:
        return 0

    # Get the screen DPI
    # px_per_in = QApplication.primaryScreen().logicalDotsPerInch()
    px_per_in = widget.mne.dpi

    # Convert to inches
    height_in = height_px / px_per_in

    # Convert pixels to inches
    in_per_V = height_in / height_V

    # Convert inches to millimeters (or something else, but using mm in the name for
    # simplicity)
    mm_per_in = _unit_per_inch[units]
    mm_per_V = in_per_V * mm_per_in
    return mm_per_V


def _convert_physical_units(value, from_unit=None, to_unit=None):
    """Convert a value from one physical unit to another."""
    if from_unit not in _unit_per_inch or to_unit not in _unit_per_inch:
        raise ValueError("Invalid units. Please use 'mm', 'cm', or 'inch'.")

    # Convert the value to inches first
    value_in_inches = value / _unit_per_inch[from_unit]

    # Convert the value from inches to the target unit
    converted_value = value_in_inches * _unit_per_inch[to_unit]

    return converted_value


def _get_channel_scaling(widget, ch_type):
    """Get channel scaling."""
    scaler = 1 if widget.mne.butterfly else 2
    inv_norm = (
        scaler
        * widget.mne.scalings[ch_type]
        * widget.mne.unit_scalings[ch_type]
        / widget.mne.scale_factor
    )
    return inv_norm
