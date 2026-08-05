# License: BSD-3-Clause
# Copyright the MNE Qt Browser contributors.

import warnings
from time import perf_counter

import mne
import numpy as np
import pytest
from mne.utils import check_version
from numpy.testing import assert_allclose, assert_array_equal
from qtpy.QtCore import Qt
from qtpy.QtTest import QTest

from mne_qt_browser._colors import _oklab_to_rgb, _rgb_to_oklab
from mne_qt_browser._utils import _disconnect

LESS_TIME = "Show fewer time points"
MORE_TIME = "Show more time points"
FEWER_CHANNELS = "Show fewer channels"
MORE_CHANNELS = "Show more channels"
REDUCE_AMPLITUDE = "Reduce amplitude"
INCREASE_AMPLITUDE = "Increase amplitude"
TOGGLE_ANNOTATIONS = "Toggle annotations mode"
SHOW_PROJECTORS = "Show projectors"


def _assert_n_figs(pg_backend, want, timeout=5.0):
    """Assert the number of visible windows, waiting for Qt to get there.

    Showing and destroying dialogs happens in the event loop, so the count only
    settles some time after the triggering action. Polling until it matches
    keeps the slow-machine headroom of a long fixed wait without paying for it
    on a fast one (``_get_n_figs`` waits 100 ms per call).
    """
    t0 = perf_counter()
    while (got := pg_backend._get_n_figs()) != want and perf_counter() - t0 < timeout:
        pass
    assert got == want


def test_disconnect_warning_filter():
    """Test that only known PySide disconnect warnings are suppressed."""

    class Signal:
        def __init__(self, message):
            self.message = message

        def disconnect(self):
            warnings.warn(self.message, RuntimeWarning)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _disconnect(
            Signal('libpyside: Failed to disconnect (None) from signal "triggered()".')
        )

    with pytest.warns(RuntimeWarning, match="unrelated warning"):
        _disconnect(Signal("unrelated warning"))


def test_splash(raw_orig, pg_backend):
    """Test that the splash screen is torn down with the browser that owns it."""
    # Every other test here passes splash=False: creating, showing and closing the
    # splash costs about a second per browser, which is most of what plotting one
    # takes. This test keeps that path (and _safe_splash) covered.
    fig = raw_orig.plot(splash=True)
    assert not hasattr(fig.mne, "splash")  # closed and dropped by _safe_splash
    _assert_n_figs(pg_backend, 1)  # the browser, with no splash left behind


def test_annotations_single_sample(raw_orig, pg_backend):
    """Test anotations with duration of 0 s."""
    # Crop and resample to avoid failing tests due to rounding in browser
    # Resampling also significantly speeds up the tests
    raw_orig = raw_orig.copy().crop(tmax=20.0).resample(100)
    # Add first annotation to initialize the description "A"
    onset = 2
    duration = 1
    description = "A"
    first_time = raw_orig.first_time
    raw_orig.annotations.append(onset + first_time, duration, description)
    duration = raw_orig.n_times / raw_orig.info["sfreq"]
    fig = raw_orig.plot(duration=duration, splash=False)
    fig.test_mode = True
    # Activate annotation_mode
    fig._fake_keypress("a")

    # Select Annotation
    fig._fake_click((2.5, 1.0), xform="data")
    # Assert that annotation was selected
    annot_dock = fig.mne.fig_annotation
    assert annot_dock.start_bx.value() == 2
    assert annot_dock.stop_bx.value() == 3

    # Test by setting values with Spinboxes
    # First, test zero duration annotation at recording start.
    annot_dock.start_bx.setValue(0)
    annot_dock.start_bx.editingFinished.emit()
    annot_dock.stop_bx.setValue(0)
    annot_dock.stop_bx.editingFinished.emit()
    # Assert that annotation starts and ends at 0 and duration is 0
    assert_allclose(raw_orig.annotations.onset[0], 0 + first_time, atol=1e-4)
    assert_allclose(raw_orig.annotations.duration[0], 0, atol=1e-4)

    # Now test zero duration annotation at arbitrary time.
    sample_time = raw_orig.times[10]
    annot_dock.stop_bx.setValue(sample_time)
    annot_dock.stop_bx.editingFinished.emit()
    annot_dock.start_bx.setValue(sample_time)
    annot_dock.start_bx.editingFinished.emit()
    # Assert that annotation starts and ends at selected time and duration is 0
    assert_allclose(raw_orig.annotations.onset[0], sample_time + first_time, atol=1e-4)
    assert_allclose(raw_orig.annotations.duration[0], 0, atol=1e-4)

    # Finally, test zero duration annotation at recording end.
    last_time = raw_orig.times[-1]
    annot_dock.stop_bx.setValue(last_time)
    annot_dock.stop_bx.editingFinished.emit()
    annot_dock.start_bx.setValue(last_time)
    annot_dock.start_bx.editingFinished.emit()
    # Assert that annotation starts and ends at last sample and duration is 0
    assert_allclose(raw_orig.annotations.onset[0], last_time + first_time, atol=1e-4)
    assert_allclose(raw_orig.annotations.duration[0], 0, atol=1e-4)


def test_annotations_recording_end(raw_orig, pg_backend):
    """Test anotations at the end of recording."""
    # Crop and resample to avoid failing tests due to rounding in browser
    # Resampling also significantly speeds up the tests
    raw_orig = raw_orig.copy().crop(tmax=20.0).resample(100)
    # Add first annotation to initialize the description "A"
    onset = 2
    duration = 1
    description = "A"
    first_time = raw_orig.first_time
    raw_orig.annotations.append(onset + first_time, duration, description)
    n_anns = len(raw_orig.annotations)
    duration = raw_orig.n_times / raw_orig.info["sfreq"]
    fig = raw_orig.plot(duration=duration, splash=False)
    fig.test_mode = True
    # Activate annotation_mode
    fig._fake_keypress("a")

    # Draw additional annotation that extends to the end of the current view
    fig._fake_click(
        (0.0, 1.0),
        add_points=[(1.0, 1.0)],
        xform="ax",
        button=1,
        kind="drag",
    )
    # Assert number of annotations did not change
    assert len(raw_orig.annotations) == n_anns
    new_annot_end = raw_orig.annotations.onset[0] + raw_orig.annotations.duration[0]
    # Assert that the annotation end extends 1 sample above the recording
    assert_allclose(
        new_annot_end,
        raw_orig.times[-1] + first_time + 1 / raw_orig.info["sfreq"],
        atol=1e-4,
    )


def test_annotations_interactions(raw_orig, pg_backend):
    """Test interactions specific to pyqtgraph-backend."""
    # Copy to avoid mutating the session-scoped fixture
    raw_orig = raw_orig.copy()
    # Add test-annotations
    onsets = np.arange(2, 8, 2) + raw_orig.first_time
    durations = np.repeat(1, len(onsets))
    descriptions = ["A", "B", "C"]
    for onset, duration, description in zip(onsets, durations, descriptions):
        raw_orig.annotations.append(onset, duration, description)
    n_anns = len(raw_orig.annotations)
    fig = raw_orig.plot(splash=False)
    fig.test_mode = True
    annot_dock = fig.mne.fig_annotation

    # Activate annotation_mode
    fig._fake_keypress("a")

    # Set current description to index 1
    annot_dock.description_cmbx.setCurrentIndex(1)
    assert fig.mne.current_description == "B"

    # Draw additional annotation
    fig._fake_click(
        (8.0, 1.0), add_points=[(9.0, 1.0)], xform="data", button=1, kind="drag"
    )
    assert len(raw_orig.annotations.onset) == n_anns + 1
    assert len(raw_orig.annotations.duration) == n_anns + 1
    assert len(raw_orig.annotations.description) == n_anns + 1
    assert raw_orig.annotations.description[-1] == "B"

    # Test remove all regions description
    annot_dock._remove_description("B")
    assert len(raw_orig.annotations.onset) == n_anns - 1
    assert len(raw_orig.annotations.duration) == n_anns - 1
    assert len(raw_orig.annotations.description) == n_anns - 1
    assert fig.mne.current_description == "A"
    assert fig.mne.selected_region is None

    # Redraw annotation (now with 'A')
    fig._fake_click(
        (4.0, 1.0), add_points=[(5.0, 1.0)], xform="data", button=1, kind="drag"
    )
    assert len(raw_orig.annotations.onset) == n_anns
    assert len(raw_orig.annotations.duration) == n_anns
    assert len(raw_orig.annotations.description) == n_anns

    # Test editing descriptions (all)
    annot_dock._edit_description_all("D")
    assert len(np.where(raw_orig.annotations.description == "D")[0]) == 2

    # Test editing descriptions (selected)
    # Select second region
    fig._fake_click((4.5, 1.0), xform="data")
    assert fig.mne.selected_region.description == "D"
    annot_dock._edit_description_selected("E")
    assert raw_orig.annotations.description[1] == "E"

    # Test Spinbox behaviour
    # Update of Spinboxes
    fig._fake_click((2.5, 1.0), xform="data")
    assert annot_dock.start_bx.value() == 2.0
    assert annot_dock.stop_bx.value() == 3.0

    # Setting values with Spinboxex
    annot_dock.start_bx.setValue(1.5)
    annot_dock.start_bx.editingFinished.emit()
    annot_dock.stop_bx.setValue(3.5)
    annot_dock.stop_bx.editingFinished.emit()
    assert raw_orig.annotations.onset[0] == 1.5 + raw_orig.first_time
    assert raw_orig.annotations.duration[0] == 2.0

    # Test SpinBox Warning
    annot_dock.start_bx.setValue(6)
    annot_dock.start_bx.editingFinished.emit()
    assert fig.msg_box.isVisible()
    assert fig.msg_box.informativeText() == "Start can't be bigger than Stop!"
    fig.msg_box.close()

    # Test that dragging annotation onto the tail of another works
    annot_dock._remove_description("E")
    annot_dock._remove_description("C")
    fig._fake_click(
        (4.0, 1.0), add_points=[(6.0, 1.0)], xform="data", button=1, kind="drag"
    )
    fig._fake_click(
        (4.0, 1.0), add_points=[(3.0, 1.0)], xform="data", button=1, kind="drag"
    )
    assert len(raw_orig.annotations.onset) == 1
    assert len(fig.mne.regions) == 1

    # Make a smaller annotation and put it into the larger one
    fig._fake_click(
        (8.0, 1.0), add_points=[(8.1, 1.0)], xform="data", button=1, kind="drag"
    )
    fig._fake_click(
        (8.0, 1.0), add_points=[(4.0, 1.0)], xform="data", button=1, kind="drag"
    )
    assert len(raw_orig.annotations.onset) == 1
    assert len(fig.mne.regions) == 1


def test_ch_specific_annot(raw_orig, pg_backend):
    """Test plotting channel specific annotations."""
    # Copy to avoid mutating the session-scoped fixture
    raw_orig = raw_orig.copy()
    ch_names = ["MEG 0133", "MEG 0142", "MEG 0143", "MEG 0423"]
    annot_onset, annot_dur = 1, 2
    annots = mne.Annotations(
        [annot_onset], [annot_dur], "some_chs", ch_names=[ch_names]
    )
    raw_orig.set_annotations(annots)

    ch_names.pop(-1)  # don't plot the last one!
    fig = raw_orig.plot(picks=ch_names, splash=False)  # omit the first one
    fig_ch_names = list(fig.mne.ch_names[fig.mne.ch_order])
    fig.test_mode = True
    annot_dock = fig.mne.fig_annotation

    # one FillBetweenItem for each channel in a channel specific annot
    annot = fig.mne.regions[0]
    assert (
        len(annot.single_channel_annots) == 4  # we still make them even for invisible
    )  # 4 channels in annots[0].single_channel_annots

    # check that a channel specific annot is plotted at the correct ypos
    which_name = raw_orig.annotations.ch_names[0][-2]
    single_channel_annot = annot.single_channel_annots[which_name]
    # the +1 is needed because ypos indexing of the traces starts at 1, not 0
    want_index = fig_ch_names.index(which_name) + 1
    got_index = np.mean(single_channel_annot.ypos).astype(int)
    assert got_index == want_index  # should be 28

    fig._fake_keypress("a")  # activate annotation mode
    # make sure our annotation is selected
    fig._fake_click((annot_onset + annot_dur / 2, 1.0), xform="data")
    assert fig.mne.current_description == "some_chs"

    # change the stop value of the annotation
    annot_dock.stop_bx.setValue(6)
    annot_dock.stop_bx.editingFinished.emit()
    # does the single channel annot stay within the annot
    assert annot_dock.stop_bx.value() == 6
    assert single_channel_annot.lower.xData[1] == 6

    # now change the start value of the annotation
    annot_dock.start_bx.setValue(4)
    annot_dock.start_bx.editingFinished.emit()
    # does the channel specific rectangle stay in sync with the annot?
    assert annot_dock.start_bx.value() == 4
    assert single_channel_annot.lower.xData[0] == 4

    ch_index = np.mean(annot.single_channel_annots["MEG 0133"].ypos).astype(int)

    # MNE >= 1.8
    if check_version("mne", "1.8"):
        # test if shift click an existing annotation removes object
        fig._fake_click(
            (4 + 2 / 2, ch_index),
            xform="data",
            button=1,
            modifier=Qt.ShiftModifier,
        )
        assert "MEG 0133" not in annot.single_channel_annots.keys()

        # test if shift click on channel adds annotation
        fig._fake_click(
            (4 + 2 / 2, ch_index),
            xform="data",
            button=1,
            modifier=Qt.ShiftModifier,
        )
        assert "MEG 0133" in annot.single_channel_annots.keys()

        # Check that channel specific annotations do not merge
        fig._fake_click(
            (2.0, 1.0), add_points=[(3.0, 1.0)], xform="data", button=1, kind="drag"
        )
        with pytest.warns(RuntimeWarning, match="combine channel-based"):
            fig._fake_click(
                (2.1, 1.0), add_points=[(5.0, 1.0)], xform="data", button=1, kind="drag"
            )

    else:
        # emit a warning if the user tries to test single channel annots
        with pytest.warns(RuntimeWarning, match="updated"):
            fig._fake_click(
                (4 + 2 / 2, ch_index),
                xform="data",
                button=1,
                modifier=Qt.ShiftModifier,
            )
            assert "MEG 0133" not in annot.single_channel_annots.keys()

    fig.close()


def test_pg_settings_dialog(raw_orig, pg_backend):
    """Test Settings Dialog toggle on/off for pyqtgraph-backend."""
    fig = raw_orig.plot(splash=False)
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    assert fig.mne.fig_settings is None
    with pytest.raises(ValueError, match="FooAction"):
        fig._fake_click_on_toolbar_action("FooAction", wait_after=0)
    for _ in range(2):
        fig._fake_click_on_toolbar_action("Settings", wait_after=0)
        assert fig.mne.fig_settings is not None
        _assert_n_figs(pg_backend, 2)
        fig._fake_click_on_toolbar_action("Settings", wait_after=0)
        assert fig.mne.fig_settings is None
        _assert_n_figs(pg_backend, 1)

    fig._fake_click_on_toolbar_action("Settings", wait_after=0)
    assert fig.mne.fig_settings is not None
    downsampling_control = fig.mne.fig_settings.downsampling_box
    assert downsampling_control.value() == fig.mne.downsampling

    downsampling_control.setValue(2)
    assert downsampling_control.value() == 2
    assert downsampling_control.value() == fig.mne.downsampling

    # Could be 6008 or 6006 depending on if MNE-Python has
    # https://github.com/mne-tools/mne-qt-browser/pull/320 (1.10+)
    allowed = (6006, 6007, 6008)
    ds = 17
    assert fig.mne.data.shape[1] in allowed
    # does not evenly divide into the data length
    assert all(x % ds != 0 for x in allowed)
    downsampling_control.setValue(ds)
    assert downsampling_control.value() == ds
    assert downsampling_control.value() == fig.mne.downsampling

    downsampling_method_control = fig.mne.fig_settings.ds_method_cmbx
    assert fig.mne.ds_method == downsampling_method_control.currentText()

    for ds_method in ("mean", "subsample", "peak"):
        downsampling_method_control.setCurrentText(ds_method)
        assert downsampling_method_control.currentText() == ds_method
        assert fig.mne.ds_method == ds_method
        fig._redraw(update_data=True)  # make sure it works
        assert fig.mne.data.shape[-1] == len(fig.mne.times)

    downsampling_method_control.setCurrentText("invalid_method_name")
    assert downsampling_method_control.currentText() != "invalid_method_name"

    sensitivity_control = fig.mne.fig_settings.scroll_sensitivity_slider
    assert fig.mne.scroll_sensitivity == sensitivity_control.value()

    sensitivity_control.setValue(100)
    assert sensitivity_control.value() == 100
    assert fig.mne.scroll_sensitivity == 100

    sensitivity_values = list(
        range(sensitivity_control.minimum(), sensitivity_control.maximum() + 1, 40)
    )
    if sensitivity_values[-1] != sensitivity_control.maximum():
        sensitivity_values.append(sensitivity_control.maximum())

    # Both sweep directions, since the slider and mne must stay in sync either way
    for values in (sensitivity_values, sensitivity_values[::-1]):
        sensitivities_mne = list()
        sensitivities_control = list()
        for val in values:
            sensitivity_control.setValue(val)
            sensitivities_mne.append(fig.mne.scroll_sensitivity)
            sensitivities_control.append(sensitivity_control.value())
        assert sensitivities_mne == values
        assert sensitivities_control == values

    # Make sure there are correct number of scaling spinboxes
    ordered_types = fig.mne.ch_types[fig.mne.ch_order]
    unique_types = np.unique(ordered_types)
    unique_types = [
        ch_type for ch_type in unique_types if ch_type in fig.mne.unit_scalings.keys()
    ]
    n_unique_types = len(unique_types)
    assert n_unique_types == len(fig.mne.fig_settings.ch_scaling_spinboxes)

    # Check that scaling spinbox has correct/expected value
    ch_type_test = unique_types[0]
    ch_spinbox = fig.mne.fig_settings.ch_scaling_spinboxes[ch_type_test]
    inv_norm = (
        fig.mne.scalings[ch_type_test]
        * fig.mne.unit_scalings[ch_type_test]
        * 2  # values multiplied by two for raw data
        / fig.mne.scale_factor
    )
    assert inv_norm == ch_spinbox.value()

    # Check that changing scaling values changes sensitivity values
    ch_scale_spinbox = fig.mne.fig_settings.ch_scaling_spinboxes[ch_type_test]
    ch_sens_spinbox = fig.mne.fig_settings.ch_sensitivity_spinboxes[ch_type_test]
    scaling_spinbox_value = ch_spinbox.value()
    sensitivity_spinbox_value = ch_sens_spinbox.value()
    scaling_value = fig.mne.scalings[ch_type_test]
    new_scaling_spinbox_value = scaling_spinbox_value * 2
    new_expected_sensitivity_spinbox_value = sensitivity_spinbox_value * 2
    ch_scale_spinbox.setValue(new_scaling_spinbox_value)
    new_scaling_value = fig.mne.scalings[ch_type_test]
    assert scaling_value != new_scaling_value
    np.testing.assert_allclose(
        ch_sens_spinbox.value(), new_expected_sensitivity_spinbox_value, atol=0.1
    )

    # Changing sensitivity values changes scaling values
    ch_scale_spinbox = fig.mne.fig_settings.ch_scaling_spinboxes[ch_type_test]
    ch_sens_spinbox = fig.mne.fig_settings.ch_sensitivity_spinboxes[ch_type_test]
    scaling_spinbox_value = ch_spinbox.value()
    sensitivity_spinbox_value = ch_sens_spinbox.value()
    scaling_value = fig.mne.scalings[ch_type_test]
    new_sensitivity_spinbox_value = sensitivity_spinbox_value * 2
    new_expected_scaling_spinbox_value = scaling_spinbox_value * 2
    ch_sens_spinbox.setValue(new_sensitivity_spinbox_value)
    assert scaling_value != fig.mne.scalings[ch_type_test]
    np.testing.assert_allclose(
        ch_scale_spinbox.value(),
        new_expected_scaling_spinbox_value,
        atol=new_expected_scaling_spinbox_value * 0.05,
    )

    # Monitor dimension update changes sensitivity values and dpi
    orig_mon_height = fig.mne.fig_settings.mon_height_spinbox.value()
    orig_mon_width = fig.mne.fig_settings.mon_width_spinbox.value()
    orig_mon_dpi = fig.mne.fig_settings.dpi_spinbox.value()
    orig_sens = ch_sens_spinbox.value()
    fig.mne.fig_settings.mon_height_spinbox.setValue(orig_mon_height / 2)
    QTest.keyPress(fig.mne.fig_settings.mon_height_spinbox.lineEdit(), Qt.Key_Return)
    fig.mne.fig_settings.mon_width_spinbox.setValue(orig_mon_width / 2)
    QTest.keyPress(fig.mne.fig_settings.mon_width_spinbox.lineEdit(), Qt.Key_Return)
    assert ch_sens_spinbox.value() != orig_sens

    # Monitor settings reset button works
    fig.mne.fig_settings._reset_monitor_spinboxes()
    assert fig.mne.fig_settings.mon_height_spinbox.value() == orig_mon_height
    assert fig.mne.fig_settings.mon_width_spinbox.value() == orig_mon_width
    assert fig.mne.fig_settings.dpi_spinbox.value() == orig_mon_dpi
    assert ch_sens_spinbox.value() == orig_sens

    # Monitor unit dropdown works (go from cm to mm or vice-versa)
    mon_unit_cmbx = fig.mne.fig_settings.mon_units_cmbx
    mon_unit_cmbx.setCurrentText("mm")
    mm_mon_height = fig.mne.fig_settings.mon_height_spinbox.value()
    mm_mon_width = fig.mne.fig_settings.mon_width_spinbox.value()
    mon_unit_cmbx.setCurrentText("cm")
    np.testing.assert_allclose(
        fig.mne.fig_settings.mon_height_spinbox.value(), mm_mon_height / 10, atol=0.1
    )
    np.testing.assert_allclose(
        fig.mne.fig_settings.mon_width_spinbox.value(), mm_mon_width / 10, atol=0.1
    )

    # Window resize changes sensitivity values
    orig_sens = ch_sens_spinbox.value()
    orig_window_size = fig.size()
    fig.resize(orig_window_size.width() * 2, orig_window_size.height() * 2)
    assert ch_sens_spinbox.value() != orig_sens


def test_pg_help_dialog(raw_orig, pg_backend):
    """Test Settings Dialog toggle on/off for pyqtgraph-backend."""
    fig = raw_orig.plot(splash=False)
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    assert fig.mne.fig_help is None
    for _ in range(2):
        fig._fake_click_on_toolbar_action("Help", wait_after=0)
        assert fig.mne.fig_help is not None
        _assert_n_figs(pg_backend, 2)
        fig._fake_click_on_toolbar_action("Help", wait_after=0)
        assert fig.mne.fig_help is None
        _assert_n_figs(pg_backend, 1)


def test_pg_toolbar_time_plus_minus(raw_orig, pg_backend):
    """Test time controls."""
    fig = raw_orig.plot(splash=False)
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    assert pg_backend._get_n_figs() == 1

    min_duration = 3 * np.diff(fig.mne.inst.times[:2])[0]  # hard code.
    xmin, xmax = fig.mne.viewbox.viewRange()[0]
    for _ in range(100):
        if xmax - xmin <= min_duration:
            break
        fig._fake_click_on_toolbar_action(LESS_TIME, wait_after=0)
        xmin, xmax = fig.mne.viewbox.viewRange()[0]
    assert xmax - xmin == min_duration

    eps = 0.01
    step = 0.25
    fig._fake_click_on_toolbar_action(MORE_TIME, wait_after=0)
    xmin_new, xmax_new = fig.mne.viewbox.viewRange()[0]
    assert xmax_new - (xmax + (xmax - xmin * step)) < eps

    xmin, xmax = fig.mne.viewbox.viewRange()[0]
    for _ in range(100):
        if xmax + fig.mne.duration * step >= fig.mne.xmax:
            break
        fig._fake_click_on_toolbar_action(MORE_TIME, wait_after=0)
        xmin, xmax = fig.mne.viewbox.viewRange()[0]

    fig._fake_click_on_toolbar_action(MORE_TIME, wait_after=0)
    fig._fake_click_on_toolbar_action(MORE_TIME, wait_after=0)

    xmin, xmax = fig.mne.viewbox.viewRange()[0]
    fig._fake_click_on_toolbar_action(MORE_TIME, wait_after=0)
    xmin_new, xmax_new = fig.mne.viewbox.viewRange()[0]
    assert xmax_new == xmax  # no effect after span maxed

    step = -0.2
    xmin, xmax = fig.mne.viewbox.viewRange()[0]
    fig._fake_click_on_toolbar_action(LESS_TIME, wait_after=0)
    xmin_new, xmax_new = fig.mne.viewbox.viewRange()[0]
    assert xmax_new == xmax + ((xmax - xmin) * step)

    xmin, xmax = fig.mne.viewbox.viewRange()[0]
    fig._fake_click_on_toolbar_action(LESS_TIME, wait_after=0)
    xmin_new, xmax_new = fig.mne.viewbox.viewRange()[0]
    assert xmax_new == xmax + ((xmax - xmin) * step)

    for _ in range(7):
        fig._fake_click_on_toolbar_action(LESS_TIME, wait_after=0)

    assert pg_backend._get_n_figs() == 1  # still alive


def test_pg_toolbar_channels_plus_minus(raw_orig, pg_backend):
    """Test channel controls."""
    fig = raw_orig.plot(splash=False)
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    assert pg_backend._get_n_figs() == 1

    # changing the number of channels should have no effect in butterfly mode
    if fig.mne.butterfly is not True:
        fig._fake_keypress("b")  # toggle butterfly mode
    fig._fake_click_on_toolbar_action(FEWER_CHANNELS, wait_after=0)
    ymin, ymax = fig.mne.viewbox.viewRange()[1]
    fig._fake_click_on_toolbar_action(FEWER_CHANNELS, wait_after=0)
    assert [ymin, ymax] == fig.mne.viewbox.viewRange()[1]
    fig._fake_click_on_toolbar_action(MORE_CHANNELS, wait_after=0)
    assert [ymin, ymax] == fig.mne.viewbox.viewRange()[1]

    if fig.mne.butterfly is True:
        fig._fake_keypress("b")  # toggle butterfly off

    for _ in range(19):  # reduce number of channels from 20 to 1
        fig._fake_click_on_toolbar_action(FEWER_CHANNELS, wait_after=0)
        ymin, ymax = fig.mne.viewbox.viewRange()[1]
    assert ymax - ymin == 2  # exactly 1 channel visible
    fig._fake_click_on_toolbar_action(FEWER_CHANNELS, wait_after=0)  # no effect
    ymin, ymax = fig.mne.viewbox.viewRange()[1]
    assert ymax - ymin == 2

    # show one more channel at a time
    fig._fake_click_on_toolbar_action(MORE_CHANNELS, wait_after=0)
    _, ymax_new = fig.mne.viewbox.viewRange()[1]
    assert ymax_new == ymax + 1

    ymin, ymax = fig.mne.viewbox.viewRange()[1]
    fig._fake_click_on_toolbar_action(MORE_CHANNELS, wait_after=0)
    _, ymax_new = fig.mne.viewbox.viewRange()[1]
    assert ymax_new == ymax + 1

    ymin, ymax = fig.mne.viewbox.viewRange()[1]
    fig._fake_click_on_toolbar_action(MORE_CHANNELS, wait_after=0)
    _, ymax_new = fig.mne.viewbox.viewRange()[1]
    assert ymax_new == ymax + 1

    assert pg_backend._get_n_figs() == 1  # still alive


def test_pg_toolbar_zoom(raw_orig, pg_backend):
    """Test zoom."""
    fig = raw_orig.plot(splash=False)
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    assert pg_backend._get_n_figs() == 1

    step = 4 / 5
    scale_factor = fig.mne.scale_factor
    fig._fake_click_on_toolbar_action(REDUCE_AMPLITUDE, wait_after=0)
    scale_factor_new = fig.mne.scale_factor
    assert scale_factor_new == scale_factor * step

    for _ in range(6):
        fig._fake_click_on_toolbar_action(REDUCE_AMPLITUDE, wait_after=0)

    step = 5 / 4
    scale_factor = fig.mne.scale_factor
    fig._fake_click_on_toolbar_action(INCREASE_AMPLITUDE, wait_after=0)
    scale_factor_new = fig.mne.scale_factor
    assert scale_factor_new == scale_factor * step

    for _ in range(6):
        fig._fake_click_on_toolbar_action(INCREASE_AMPLITUDE, wait_after=0)

    assert pg_backend._get_n_figs() == 1  # still alive


def test_pg_toolbar_annotations(raw_orig, pg_backend):
    """Test annotations mode."""
    fig = raw_orig.plot(splash=False)
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    assert pg_backend._get_n_figs() == 1

    state_annotation_widget = fig.mne.annotation_mode
    for _ in range(4):
        fig._fake_click_on_toolbar_action(TOGGLE_ANNOTATIONS, wait_after=0)
        assert fig.mne.annotation_mode != state_annotation_widget
        state_annotation_widget = fig.mne.annotation_mode

    _assert_n_figs(pg_backend, 1)  # still alive


def test_pg_toolbar_actions(raw_orig, pg_backend):
    """Test toolbar all actions combined.

    Toolbar actions here create a separate QDialog window.
    We test the state machine for each window toggle button.
    """
    fig = raw_orig.plot(splash=False)
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    assert pg_backend._get_n_figs() == 1

    for action, n_figs in [
        (SHOW_PROJECTORS, 2),
        ("Settings", 3),
        ("Settings", 2),
        ("Help", 3),
        ("Settings", 4),
        (SHOW_PROJECTORS, 3),
        ("Settings", 2),
        ("Help", 1),
    ]:
        fig._fake_click_on_toolbar_action(action, wait_after=0)
        _assert_n_figs(pg_backend, n_figs)


# Oklab values taken from coloraide on 2026/07/20
@pytest.mark.parametrize(
    "rgb, lab",
    [
        [(0, 0, 1), (0.45201372, -0.03245698, -0.31152817)],  # blue
        [(1, 1, 1), (1, 0, 0)],  # white
        [(0, 0, 0), (0, 0, 0)],  # black
        # np.random.default_rng(0).uniform(0, 1, (4, 3))
        [
            (0.63696169, 0.26978671, 0.04097352),
            (0.50652981, 0.09724356, 0.09878626),
        ],
        [
            (0.01652764, 0.81327024, 0.91275558),
            (0.78437508, -0.11689532, -0.06835410),
        ],
        [
            (0.60663578, 0.72949656, 0.54362499),
            (0.75317526, -0.05269919, 0.05236921),
        ],
        [
            (0.93507242, 0.81585355, 0.0027385),
            (0.85722598, -0.02622580, 0.17537609),
        ],
    ],
)
def test_color_conversion(rgb, lab):
    """Test color conversions against manually run ones."""
    our_lab = _rgb_to_oklab(rgb)
    assert_allclose(our_lab, lab, atol=1e-5)
    rgb_2 = _oklab_to_rgb(lab)
    assert_allclose(rgb, rgb_2, atol=1e-5)


def test_zscore_rgba(raw_orig, pg_backend):
    """Test the z-score overview RGBA mapping."""
    fig = raw_orig.plot(splash=False)
    fig.test_mode = True
    # One symmetric ramp channel, one all-NaN channel, and one near-constant
    # channel; the latter triggers SciPy's catastrophic-cancellation
    # RuntimeWarning, which must not escape (it would kill the load thread
    # under warnings-as-errors; gh-428)
    near_constant = np.ones(5)
    near_constant[0] += 1e-15
    data = np.array([np.linspace(-2, 2, 5), [np.nan] * 5, near_constant])
    fig._get_zscore(data, max_pixel_width=5)
    zrgba = fig.mne.zscore_rgba
    assert zrgba.dtype == np.uint8
    assert zrgba.shape == (3, 5, 4)
    # Negative z-scores fade to blue, positive to red, extrema fully opaque
    expected = np.array(
        [
            [0, 0, 255, 255],
            [0, 0, 255, 127],
            [0, 0, 0, 0],
            [255, 0, 0, 127],
            [255, 0, 0, 255],
        ],
        dtype=np.uint8,
    )
    assert_allclose(zrgba[0], expected)
    # NaN channels are fully transparent
    assert_allclose(zrgba[1], 0)


def test_overview_bad_epochs_dropped(raw_orig, pg_backend):
    """Test bad-epoch rect positions after resize when epochs were dropped."""
    epochs = mne.make_fixed_length_epochs(
        raw_orig.copy().crop(tmax=20.0), duration=2.0, preload=True
    )
    epochs.drop([0, 2])  # make selection non-contiguous
    fig = epochs.plot(splash=False)
    fig.test_mode = True
    epo_num = epochs.selection[-1]
    fig.mne.bad_epochs.append(epo_num)
    overview_bar = fig.mne.overview_bar
    overview_bar.update_bad_epochs()
    assert epo_num in overview_bar.bad_epoch_rect_dict
    # Resizing must reposition the rect via the epoch index, not its number
    fig.resize(fig.width() + 30, fig.height())
    QTest.qWait(100)
    rect = overview_bar.bad_epoch_rect_dict[epo_num].rect()
    epo_idx = epochs.selection.tolist().index(epo_num)
    expected_left = overview_bar._mapFromData(fig.mne.boundary_times[epo_idx], 0).x()
    assert_allclose(rect.left(), expected_left)


def test_description_cmbx_preserves_selection(raw_orig, pg_backend):
    """Test that rebuilding the description combobox keeps the selection."""
    raw_orig = raw_orig.copy().crop(tmax=5.0)
    first_time = raw_orig.first_time
    raw_orig.annotations.append(1 + first_time, 1, "A")
    raw_orig.annotations.append(3 + first_time, 1, "B")
    fig = raw_orig.plot(splash=False)
    fig.test_mode = True
    dock = fig.mne.fig_annotation
    dock.description_cmbx.setCurrentText("B")
    assert fig.mne.current_description == "B"
    # Rebuilding emits currentIndexChanged(-1) then (0), which used to clobber
    # the current description with the first item
    dock._update_description_cmbx()
    assert dock.description_cmbx.currentText() == "B"
    assert fig.mne.current_description == "B"
    # A no-longer-existing description falls back to the first item
    fig.mne.current_description = "gone"
    dock._update_description_cmbx()
    assert dock.description_cmbx.currentText() == "A"
    assert fig.mne.current_description == "A"


def test_time_scrollbar_page_step(raw_orig, pg_backend):
    """Test that the scrollbar slider represents the visible fraction."""
    fig = raw_orig.plot(duration=10.0, splash=False)
    fig.test_mode = True
    ax_hscroll = fig.mne.ax_hscroll
    # One page in scrollbar units is the visible duration times step_factor
    assert ax_hscroll.pageStep() == int(fig.mne.duration * ax_hscroll.step_factor)
    assert ax_hscroll.pageStep() == int(fig.mne.scroll_sensitivity)


def test_qsettings_string_bools(raw_orig, pg_backend):
    """Test that boolean settings round-trip through their string form."""
    from mne_qt_browser import _pg_figure

    # On some platforms QSettings returns booleans as "true"/"false" strings
    qsettings = _pg_figure.QSettings()
    qsettings.setValue("antialiasing", "false")
    qsettings.setValue("overview_visible", "true")
    qsettings.sync()
    fig = raw_orig.plot(splash=False)
    assert fig.mne.antialiasing is False
    assert fig.mne.overview_visible is True


def test_message_box_reset(raw_orig, pg_backend):
    """Test that message_box state does not leak into the next message."""
    from qtpy.QtWidgets import QMessageBox

    fig = raw_orig.plot(splash=False)
    fig.test_mode = True
    fig.message_box(
        "first",
        info_text="details",
        buttons=QMessageBox.Yes | QMessageBox.No,
        icon=QMessageBox.Critical,
    )
    fig.message_box("second")
    assert fig.msg_box.informativeText() == ""
    assert fig.msg_box.standardButtons() == QMessageBox.Ok
    assert fig.msg_box.icon() == QMessageBox.NoIcon


def test_get_onset_idx_float_tolerance(raw_orig, pg_backend):
    """Test that annotation lookup survives sub-sample float drift."""
    # raw_orig is shared across the session and other tests mutate its
    # annotations in place, so start from a clean slate here
    raw_orig = raw_orig.copy().crop(tmax=5.0)
    raw_orig.set_annotations(None)
    first_time = raw_orig.first_time
    raw_orig.annotations.append(1 + first_time, 1, "A")
    raw_orig.annotations.append(3 + first_time, 1, "B")
    fig = raw_orig.plot(splash=False)
    fig.test_mode = True
    drift = 0.1 / raw_orig.info["sfreq"]
    # Each annotation still resolves to its own index despite sub-sample drift
    assert fig._get_onset_idx(3.0 + drift) == 1  # "B"
    assert fig._get_onset_idx(1.0 - drift) == 0  # "A"


def _wait_precompute(fig):
    for _ in range(600):
        if fig.mne.data_precomputed:
            return
        QTest.qWait(100)
    raise AssertionError("Precomputation did not finish")


def _traces_drawn(fig):
    """Get what each trace draws plus its zero line, relative to its own baseline."""
    # Relative to the baseline (rather than in data coordinates) so that assert_allclose
    # is not dominated by ypos, which is the same in both modes anyway
    out = dict()
    for trace in fig.mne.traces:
        traces = [trace] + list(getattr(trace, "child_traces", []))
        # Child traces (epochs) each draw one color and NaN out the rest
        drawn = np.full(len(trace.xData), np.nan)
        for this in traces:
            y = this.transform().m22() * np.asarray(this.yData)
            drawn[np.isfinite(y)] = y[np.isfinite(y)]
        if _HAS_ZERO_LINE_OFFSET:
            zero = trace._true_zero_ypos() - trace.ypos
            drawn = np.concatenate([drawn, [zero]])
        out[trace.ch_name] = drawn
    return out


# BrowserBase only tracks the DC offset of the shown range (and thus places the zero
# line at the true zero) as of mne 1.13
_HAS_ZERO_LINE_OFFSET = check_version("mne", "1.13")


@pytest.mark.skipif(
    not check_version("mne", "1.10"),
    # ... which the precompute path never did, so the two modes cannot match there
    reason="mne < 1.10 pads the shown range with two extra samples",
)
@pytest.mark.parametrize("clipping", ("transparent", "clamp", None))
def test_precompute_matches_on_the_fly(raw_orig, pg_backend, clipping):
    """Test that precomputed data is displayed like data processed on the fly."""
    # A window that has stim events but whose stim maxima differ from the maxima over
    # the whole recording (gh-270), and all channel types visible
    raw_orig = raw_orig.copy().crop(tmax=10.0)
    kwargs = dict(
        clipping=clipping,
        n_channels=len(raw_orig.ch_names),
        duration=2.0,
        start=3.0,
        splash=False,
    )
    drawn = dict()
    for precompute in (False, True):
        fig = raw_orig.plot(precompute=precompute, **kwargs)
        fig.test_mode = True
        if precompute:
            _wait_precompute(fig)
        drawn[precompute] = [_traces_drawn(fig)]
        # Each of these has at some point been broken by precomputation only
        for action in (
            lambda: fig._toggle_all_projs(),  # projectors
            lambda: fig._click_ch_name(0, 0),  # bad channel (changes projector)
            lambda: fig._fake_keypress("="),  # scale_factor
            lambda: fig.mne.scalings.__setitem__("grad", 1e-10) or fig._redraw(),
            lambda: fig._fake_keypress("d"),  # DC removal
            lambda: fig._fake_keypress("right"),  # scroll
        ):
            action()
            QTest.qWait(50)
            if precompute:
                _wait_precompute(fig)
            drawn[precompute].append(_traces_drawn(fig))
        fig.close()

    for on_the_fly, precomputed in zip(drawn[False], drawn[True]):
        assert set(on_the_fly) == set(precomputed)
        for ch_name, expected in on_the_fly.items():
            got = precomputed[ch_name]
            # Clipping must kick in for the same samples in both modes
            assert_array_equal(np.isnan(got), np.isnan(expected), err_msg=ch_name)
            assert_allclose(got, expected, atol=1e-10, err_msg=ch_name)


def test_butterfly_scalebars(raw_orig, pg_backend):
    """Test that butterfly mode matches the matplotlib backend (gh-276)."""
    raw_orig = raw_orig.copy().crop(tmax=5.0)
    fig = raw_orig.plot()
    fig.test_mode = True
    ch_types = fig.mne.butterfly_type_order
    assert ch_types == ["grad", "mag", "eeg", "eog", "stim"]
    normal_texts = fig._get_scale_bar_texts()
    normal_scale_factor = fig.mne.scale_factor

    def _check_butterfly():
        # Each channel type gets exactly one y-unit, without extra space at the
        # top and bottom of the plot
        assert_allclose(fig.mne.viewbox.viewRange()[1], [0.5, len(ch_types) + 0.5])
        # Traces are drawn at half amplitude, so the bars span half a unit ...
        for ci, ch_type in enumerate(ch_types, 1):
            if ch_type not in fig.mne.scalebars:  # stim has no scalebar
                continue
            assert_allclose(
                fig.mne.scalebars[ch_type].get_ydata(), [ci - 0.25, ci + 0.25]
            )
        # ... which keeps the values identical to non-butterfly mode
        assert fig._get_scale_bar_texts() == normal_texts
        # Channel types plotted higher up are drawn over the ones below them
        zvalues = {tr.ch_type: tr.zValue() for tr in fig.mne.traces if not tr.isbad}
        assert [zvalues[ch_type] for ch_type in ch_types] == sorted(
            zvalues.values(), reverse=True
        )

    fig._fake_keypress("b")
    _check_butterfly()

    # Toggling back and forth is a no-op for the scale factor
    fig._fake_keypress("b")
    assert fig.mne.scale_factor == normal_scale_factor
    assert fig._get_scale_bar_texts() == normal_texts

    # Starting out in butterfly mode gives the same result
    fig = raw_orig.plot(butterfly=True)
    fig.test_mode = True
    _check_butterfly()
