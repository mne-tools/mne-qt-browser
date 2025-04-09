# Author: Martin Schulz <dev@earthman-music.de>
#
# License: BSD-3-Clause


import numpy as np
import pytest
from mne import Annotations
from mne.utils import check_version
from numpy.testing import assert_allclose
from qtpy.QtCore import Qt
from qtpy.QtTest import QTest

from mne_qt_browser._colors import _lab_to_rgb, _rgb_to_lab

LESS_TIME = "Show fewer time points"
MORE_TIME = "Show more time points"
FEWER_CHANNELS = "Show fewer channels"
MORE_CHANNELS = "Show more channels"
REDUCE_AMPLITUDE = "Reduce amplitude"
INCREASE_AMPLITUDE = "Increase amplitude"
TOGGLE_ANNOTATIONS = "Toggle annotations mode"
SHOW_PROJECTORS = "Show projectors"


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
    fig = raw_orig.plot(duration=raw_orig.duration)
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
    fig = raw_orig.plot(duration=raw_orig.duration)
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
    # Add test-annotations
    onsets = np.arange(2, 8, 2) + raw_orig.first_time
    durations = np.repeat(1, len(onsets))
    descriptions = ["A", "B", "C"]
    for onset, duration, description in zip(onsets, durations, descriptions):
        raw_orig.annotations.append(onset, duration, description)
    n_anns = len(raw_orig.annotations)
    fig = raw_orig.plot()
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
    ch_names = ["MEG 0133", "MEG 0142", "MEG 0143", "MEG 0423"]
    annot_onset, annot_dur = 1, 2
    annots = Annotations([annot_onset], [annot_dur], "some_chs", ch_names=[ch_names])
    raw_orig.set_annotations(annots)

    ch_names.pop(-1)  # don't plot the last one!
    fig = raw_orig.plot(picks=ch_names)  # omit the first one
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
    fig = raw_orig.plot()
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    QTest.qWait(50)
    assert fig.mne.fig_settings is None
    with pytest.raises(ValueError, match="FooAction"):
        fig._fake_click_on_toolbar_action("FooAction")
    fig._fake_click_on_toolbar_action("Settings", wait_after=500)
    assert fig.mne.fig_settings is not None
    assert pg_backend._get_n_figs() == 2
    fig._fake_click_on_toolbar_action("Settings", wait_after=500)
    assert pg_backend._get_n_figs() == 1
    assert fig.mne.fig_settings is None
    fig._fake_click_on_toolbar_action("Settings", wait_after=500)
    assert pg_backend._get_n_figs() == 2
    assert fig.mne.fig_settings is not None
    fig._fake_click_on_toolbar_action("Settings", wait_after=500)
    assert pg_backend._get_n_figs() == 1
    assert fig.mne.fig_settings is None

    fig._fake_click_on_toolbar_action("Settings", wait_after=500)
    assert fig.mne.fig_settings is not None
    downsampling_control = fig.mne.fig_settings.downsampling_box
    assert downsampling_control.value() == fig.mne.downsampling

    downsampling_control.setValue(2)
    QTest.qWait(100)
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
    QTest.qWait(100)
    assert downsampling_control.value() == ds
    assert downsampling_control.value() == fig.mne.downsampling

    QTest.qWait(100)
    downsampling_method_control = fig.mne.fig_settings.ds_method_cmbx
    assert fig.mne.ds_method == downsampling_method_control.currentText()

    downsampling_method_control.setCurrentText("mean")
    QTest.qWait(100)
    assert downsampling_method_control.currentText() == "mean"
    assert fig.mne.ds_method == "mean"
    fig._redraw(update_data=True)  # make sure it works
    assert fig.mne.data.shape[-1] == len(fig.mne.times)

    downsampling_method_control.setCurrentText("subsample")
    QTest.qWait(100)
    assert downsampling_method_control.currentText() == "subsample"
    assert fig.mne.ds_method == "subsample"
    fig._redraw(update_data=True)  # make sure it works
    assert fig.mne.data.shape[-1] == len(fig.mne.times)

    downsampling_method_control.setCurrentText("peak")
    QTest.qWait(100)
    assert downsampling_method_control.currentText() == "peak"
    assert fig.mne.ds_method == "peak"
    fig._redraw(update_data=True)  # make sure it works
    assert fig.mne.data.shape[-1] == len(fig.mne.times)

    downsampling_method_control.setCurrentText("invalid_method_name")
    QTest.qWait(100)
    assert downsampling_method_control.currentText() != "invalid_method_name"

    sensitivity_control = fig.mne.fig_settings.scroll_sensitivity_slider
    assert fig.mne.scroll_sensitivity == sensitivity_control.value()

    sensitivity_control.setValue(100)
    QTest.qWait(100)
    assert sensitivity_control.value() == 100
    assert fig.mne.scroll_sensitivity == 100

    QTest.qWait(100)
    sensitivity_values = list(
        range(sensitivity_control.minimum(), sensitivity_control.maximum() + 1, 40)
    )
    if sensitivity_values[-1] != sensitivity_control.maximum():
        sensitivity_values.append(sensitivity_control.maximum())

    sensitivities_mne = list()
    sensitivities_control = list()
    for val in sensitivity_values:
        sensitivity_control.setValue(val)
        QTest.qWait(50)
        sensitivities_mne.append(fig.mne.scroll_sensitivity)
        sensitivities_control.append(sensitivity_control.value())
    assert sensitivities_mne == sensitivity_values
    assert sensitivities_control == sensitivity_values

    sensitivity_values = sensitivity_values[::-1]
    sensitivities_mne = list()
    sensitivities_control = list()
    for val in sensitivity_values:
        sensitivity_control.setValue(val)
        QTest.qWait(50)
        sensitivities_mne.append(fig.mne.scroll_sensitivity)
        sensitivities_control.append(sensitivity_control.value())
    assert sensitivities_mne == sensitivity_values
    assert sensitivities_control == sensitivity_values

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
    fig = raw_orig.plot()
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    QTest.qWait(50)
    assert fig.mne.fig_help is None
    fig._fake_click_on_toolbar_action("Help", wait_after=500)
    assert fig.mne.fig_help is not None
    assert pg_backend._get_n_figs() == 2
    fig._fake_click_on_toolbar_action("Help", wait_after=500)
    assert fig.mne.fig_help is None
    assert pg_backend._get_n_figs() == 1
    fig._fake_click_on_toolbar_action("Help", wait_after=500)
    assert fig.mne.fig_help is not None
    assert pg_backend._get_n_figs() == 2
    fig._fake_click_on_toolbar_action("Help", wait_after=500)
    assert fig.mne.fig_help is None
    assert pg_backend._get_n_figs() == 1


def test_pg_toolbar_time_plus_minus(raw_orig, pg_backend):
    """Test time controls."""
    fig = raw_orig.plot()
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    assert pg_backend._get_n_figs() == 1

    min_duration = 3 * np.diff(fig.mne.inst.times[:2])[0]  # hard code.
    xmin, xmax = fig.mne.viewbox.viewRange()[0]
    for _ in range(100):
        if xmax - xmin <= min_duration:
            break
        fig._fake_click_on_toolbar_action(LESS_TIME, wait_after=20)
        xmin, xmax = fig.mne.viewbox.viewRange()[0]
    assert xmax - xmin == min_duration

    eps = 0.01
    step = 0.25
    fig._fake_click_on_toolbar_action(MORE_TIME, wait_after=100)
    xmin_new, xmax_new = fig.mne.viewbox.viewRange()[0]
    assert xmax_new - (xmax + (xmax - xmin * step)) < eps

    xmin, xmax = fig.mne.viewbox.viewRange()[0]
    for _ in range(100):
        if xmax + fig.mne.duration * step >= fig.mne.xmax:
            break
        fig._fake_click_on_toolbar_action(MORE_TIME, wait_after=20)
        xmin, xmax = fig.mne.viewbox.viewRange()[0]

    fig._fake_click_on_toolbar_action(MORE_TIME, wait_after=200)
    fig._fake_click_on_toolbar_action(MORE_TIME, wait_after=200)

    xmin, xmax = fig.mne.viewbox.viewRange()[0]
    fig._fake_click_on_toolbar_action(MORE_TIME, wait_after=200)
    xmin_new, xmax_new = fig.mne.viewbox.viewRange()[0]
    assert xmax_new == xmax  # no effect after span maxed

    step = -0.2
    xmin, xmax = fig.mne.viewbox.viewRange()[0]
    fig._fake_click_on_toolbar_action(LESS_TIME, wait_after=200)
    xmin_new, xmax_new = fig.mne.viewbox.viewRange()[0]
    assert xmax_new == xmax + ((xmax - xmin) * step)

    xmin, xmax = fig.mne.viewbox.viewRange()[0]
    fig._fake_click_on_toolbar_action(LESS_TIME, wait_after=200)
    xmin_new, xmax_new = fig.mne.viewbox.viewRange()[0]
    assert xmax_new == xmax + ((xmax - xmin) * step)

    for _ in range(7):
        fig._fake_click_on_toolbar_action(LESS_TIME, wait_after=20)

    assert pg_backend._get_n_figs() == 1  # still alive


def test_pg_toolbar_channels_plus_minus(raw_orig, pg_backend):
    """Test channel controls."""
    fig = raw_orig.plot()
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    assert pg_backend._get_n_figs() == 1

    if fig.mne.butterfly is not True:
        fig._fake_keypress("b")  # toggle butterfly mode
    fig._fake_click_on_toolbar_action(FEWER_CHANNELS, wait_after=100)
    ymin, ymax = fig.mne.viewbox.viewRange()[1]
    fig._fake_click_on_toolbar_action(FEWER_CHANNELS, wait_after=100)
    assert [ymin, ymax] == fig.mne.viewbox.viewRange()[1]
    fig._fake_click_on_toolbar_action(MORE_CHANNELS, wait_after=100)
    assert [ymin, ymax] == fig.mne.viewbox.viewRange()[1]

    if fig.mne.butterfly is True:
        fig._fake_keypress("b")  # toggle butterfly off

    for _ in range(10):
        if ymax - ymin <= 2:
            break
        fig._fake_click_on_toolbar_action(FEWER_CHANNELS, wait_after=40)
        ymin, ymax = fig.mne.viewbox.viewRange()[1]
    assert ymax - ymin == 2
    fig._fake_click_on_toolbar_action(FEWER_CHANNELS, wait_after=40)
    ymin, ymax = fig.mne.viewbox.viewRange()[1]
    assert ymax - ymin == 2

    step = 10
    fig._fake_click_on_toolbar_action(MORE_CHANNELS, wait_after=100)
    ymin_new, ymax_new = fig.mne.viewbox.viewRange()[1]
    assert ymax_new == ymax + step

    ymin, ymax = fig.mne.viewbox.viewRange()[1]
    fig._fake_click_on_toolbar_action(MORE_CHANNELS, wait_after=100)
    ymin_new, ymax_new = fig.mne.viewbox.viewRange()[1]
    assert ymax_new == ymax + step

    ymin, ymax = fig.mne.viewbox.viewRange()[1]
    fig._fake_click_on_toolbar_action(MORE_CHANNELS, wait_after=100)
    ymin_new, ymax_new = fig.mne.viewbox.viewRange()[1]
    assert ymax_new == ymax + step

    assert pg_backend._get_n_figs() == 1  # still alive


def test_pg_toolbar_zoom(raw_orig, pg_backend):
    """Test zoom."""
    fig = raw_orig.plot()
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    assert pg_backend._get_n_figs() == 1

    step = 4 / 5
    scale_factor = fig.mne.scale_factor
    fig._fake_click_on_toolbar_action(REDUCE_AMPLITUDE, wait_after=100)
    scale_factor_new = fig.mne.scale_factor
    assert scale_factor_new == scale_factor * step

    for _ in range(6):
        fig._fake_click_on_toolbar_action(REDUCE_AMPLITUDE, wait_after=100)

    step = 5 / 4
    scale_factor = fig.mne.scale_factor
    fig._fake_click_on_toolbar_action(INCREASE_AMPLITUDE, wait_after=100)
    scale_factor_new = fig.mne.scale_factor
    assert scale_factor_new == scale_factor * step

    for _ in range(6):
        fig._fake_click_on_toolbar_action(INCREASE_AMPLITUDE, wait_after=100)

    assert pg_backend._get_n_figs() == 1  # still alive


def test_pg_toolbar_annotations(raw_orig, pg_backend):
    """Test annotations mode."""
    fig = raw_orig.plot()
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    assert pg_backend._get_n_figs() == 1

    state_annotation_widget = fig.mne.annotation_mode
    fig._fake_click_on_toolbar_action(TOGGLE_ANNOTATIONS, wait_after=100)
    assert fig.mne.annotation_mode != state_annotation_widget

    fig._fake_click_on_toolbar_action(TOGGLE_ANNOTATIONS, wait_after=300)
    fig._fake_click_on_toolbar_action(TOGGLE_ANNOTATIONS, wait_after=300)
    fig._fake_click_on_toolbar_action(TOGGLE_ANNOTATIONS, wait_after=300)

    assert pg_backend._get_n_figs() == 1  # still alive


def test_pg_toolbar_actions(raw_orig, pg_backend):
    """Test toolbar all actions combined.

    Toolbar actions here create a separate QDialog window.
    We test the state machine for each window toggle button.
    """
    fig = raw_orig.plot()
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    assert pg_backend._get_n_figs() == 1

    fig._fake_click_on_toolbar_action(SHOW_PROJECTORS, wait_after=200)
    assert pg_backend._get_n_figs() == 2
    fig._fake_click_on_toolbar_action("Settings", wait_after=200)
    assert pg_backend._get_n_figs() == 3
    fig._fake_click_on_toolbar_action("Settings", wait_after=100)
    assert pg_backend._get_n_figs() == 2
    fig._fake_click_on_toolbar_action("Help", wait_after=200)
    assert pg_backend._get_n_figs() == 3
    fig._fake_click_on_toolbar_action("Settings", wait_after=200)
    assert pg_backend._get_n_figs() == 4
    fig._fake_click_on_toolbar_action(SHOW_PROJECTORS, wait_after=200)
    assert pg_backend._get_n_figs() == 3
    fig._fake_click_on_toolbar_action("Settings", wait_after=100)
    assert pg_backend._get_n_figs() == 2
    fig._fake_click_on_toolbar_action("Help", wait_after=100)
    assert pg_backend._get_n_figs() == 1


# LAB values taken from colorspacious on 2024/06/10
@pytest.mark.parametrize(
    "rgb, lab",
    [
        [(0, 0, 1), (32.30269787, 79.19228008, -107.86329661)],  # green
        [(1, 1, 1), (100, 0, 0)],  # white
        [(0, 0, 0), (0, 0, 0)],  # black
        # np.random.default_rng(0).uniform(0, 1, (4, 3))
        [
            (0.63696169, 0.26978671, 0.04097352),
            (41.18329695, 36.11095837, 48.52748511),
        ],
        [
            (0.01652764, 0.81327024, 0.91275558),
            (76.50078586, -33.20150417, -24.47911354),
        ],
        [
            (0.60663578, 0.72949656, 0.54362499),
            (72.17250521, -19.45430481, 20.62037424),
        ],
        [
            (0.93507242, 0.81585355, 0.0027385),
            (83.64455095, -5.45852637, 84.04513029),
        ],
    ],
)
def test_color_conversion(rgb, lab):
    """Test color conversions against manually run ones."""
    our_lab = _rgb_to_lab(rgb)
    assert_allclose(our_lab, lab, atol=2e-2)
    rgb_2 = _lab_to_rgb(lab)
    assert_allclose(rgb, rgb_2, atol=2e-2)
