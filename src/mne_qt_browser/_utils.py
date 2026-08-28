# License: BSD-3-Clause
# Copyright the MNE Qt Browser contributors.

import functools
import warnings
import weakref

import numpy as np
from mne.io.pick import _DATA_CH_TYPES_ORDER_DEFAULT
from qtpy.QtCore import QPoint
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


# Butterfly mode draws traces at half amplitude (like the matplotlib backend, gh-276),
# which equally shortens the scalebar and the value it stands for
BUTTERFLY_SCALE = 0.5


def _butterfly_scale(mne):
    """Get the factor butterfly mode shrinks traces and scalebars by."""
    return BUTTERFLY_SCALE if mne.butterfly else 1.0


def _disconnect(sig, *, allow_error=False):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "(?:libpyside: )?Failed to disconnect",
                category=RuntimeWarning,
            )
            sig.disconnect()
    except (TypeError, RuntimeError, SystemError):  # if are no connections, ignore
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


def _screen(widget):
    """Get the screen the widget is (mostly) displayed on."""
    try:
        # Qt 5.14+
        return widget.screen()
    except AttributeError:
        # Top center of the widget
        screen = QGuiApplication.screenAt(
            widget.mapToGlobal(QPoint(widget.width() // 2, 0))
        )
        if screen is None:
            screen = QGuiApplication.primaryScreen()

        return screen


def _screen_geometry(widget):
    return _screen(widget).geometry()


def _unique_ordered_ch_types(mne):
    """Get the unique channel types in displayed order."""
    ordered_types = mne.ch_types[mne.ch_order]
    unique_type_idxs = np.unique(ordered_types, return_index=True)[1]
    return [ordered_types[idx] for idx in sorted(unique_type_idxs)]


def _calc_chan_type_to_physical(widget, ch_type, units="mm"):
    """Convert data to physical units."""
    # Butterfly mode needs no correction here: a shorter scalebar standing for a
    # proportionally smaller value cancels out, leaving only its scale_factor effect
    return _get_y_unit_scaling(widget, ch_type) / _calc_data_unit_to_physical(
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


def _get_y_unit_scaling(widget, ch_type):
    """Get the data value spanned by one y-unit (i.e., by one channel's row)."""
    # data are normalized to +/-0.5 of a y-unit (see _get_display_norms), hence the 2
    return (
        2
        * widget.mne.scalings[ch_type]
        * widget.mne.unit_scalings[ch_type]
        / widget.mne.scale_factor
    )


def _get_channel_scaling(widget, ch_type):
    """Get the value a scalebar stands for."""
    return _butterfly_scale(widget.mne) * _get_y_unit_scaling(widget, ch_type)


def is_variable_duration(mne):
    """Whether the browsed epochs have trials of differing duration.

    Parameters
    ----------
    mne : object
        The browser parameter container.

    Returns
    -------
    variable : bool
        ``True`` only for ragged epochs from MNE-Python versions that support
        them; older versions have no such attribute and keep the fixed path.
    """
    return bool(getattr(getattr(mne, "inst", None), "variable_duration", False))


def epoch_window(boundary_times, start_ix, n_epochs):
    """Return the start time and duration of a window of whole epochs.

    Epochs need not share a duration, so a window of ``n_epochs`` of them spans
    whatever lies between two boundaries rather than a fixed number of seconds.
    ``start_ix`` is clamped so the requested epochs stay visible whenever the
    object is long enough to allow it.

    Parameters
    ----------
    boundary_times : array
        Cumulative epoch edges in seconds, including both ends.
    start_ix : int
        Index of the first epoch to show.
    n_epochs : int
        Number of epochs to show.

    Returns
    -------
    t_start : float
        Time of the first boundary.
    duration : float
        Seconds spanned by the requested epochs.
    """
    boundary_times = np.asarray(boundary_times, float)
    n_total = len(boundary_times) - 1
    n_epochs = int(np.clip(n_epochs, 1, n_total))
    start_ix = int(np.clip(start_ix, 0, n_total - n_epochs))
    stop_ix = start_ix + n_epochs
    return (
        float(boundary_times[start_ix]),
        float(boundary_times[stop_ix] - boundary_times[start_ix]),
    )


def epoch_index_at(boundary_times, t):
    """Return the index of the epoch containing a display time.

    Parameters
    ----------
    boundary_times : array
        Cumulative epoch edges in seconds, including both ends.
    t : float
        A time on the browser's concatenated axis.

    Returns
    -------
    idx : int
        Index of the containing epoch, clamped to the available range.
    """
    boundary_times = np.asarray(boundary_times, float)
    n_total = len(boundary_times) - 1
    return int(
        np.clip(
            np.searchsorted(boundary_times[1:], t, side="right"), 0, max(n_total - 1, 0)
        )
    )


def latency_at(boundary_times, epoch_tmins, sfreq, t):
    """Convert a display time to the latency relative to its epoch's event.

    Parameters
    ----------
    boundary_times : array
        Cumulative epoch edges in seconds, including both ends.
    epoch_tmins : array
        Start time of each epoch relative to its own event.
    sfreq : float
        Sampling frequency.
    t : float
        A time on the browser's concatenated axis.

    Returns
    -------
    latency : float
        Time relative to the event of the epoch that contains ``t``.
    """
    idx = epoch_index_at(boundary_times, t)
    offset = round((float(t) - float(boundary_times[idx])) * sfreq)
    return float(epoch_tmins[idx]) + offset / sfreq


def latency_positions(
    boundary_times, epoch_tmins, epoch_tmaxs, latency, sfreq, epoch_ixs
):
    """Return where a latency falls in each epoch that reaches it.

    Epochs shorter than ``latency`` simply do not get a position, so the result
    may be shorter than ``epoch_ixs``.

    Parameters
    ----------
    boundary_times : array
        Cumulative epoch edges in seconds, including both ends.
    epoch_tmins, epoch_tmaxs : array
        Start and end of each epoch relative to its own event.
    latency : float
        Time relative to the event, as returned by :func:`latency_at`.
    sfreq : float
        Sampling frequency.
    epoch_ixs : array-like
        Indices of the epochs to consider.

    Returns
    -------
    ixs : array
        The epochs that contain the latency.
    positions : array
        Their positions on the browser's concatenated axis.
    """
    tol = 0.5 / sfreq
    keep, xs = list(), list()
    for idx in np.atleast_1d(np.asarray(epoch_ixs, int)):
        tmin, tmax = float(epoch_tmins[idx]), float(epoch_tmaxs[idx])
        if tmin - tol <= latency <= tmax + tol:
            keep.append(idx)
            xs.append(float(boundary_times[idx]) + (latency - tmin))
    return np.array(keep, int), np.array(xs, float)
