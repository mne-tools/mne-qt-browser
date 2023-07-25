# -*- coding: utf-8 -*-
# Author: Martin Schulz <dev@earthman-music.de>
#
# License: BSD-3-Clause

import numpy as np
from qtpy.QtTest import QTest


def test_annotations_interactions(raw_orig, pg_backend):
    """Test interactions specific to pyqtgraph-backend."""
    # Add test-annotations
    onsets = np.arange(2, 8, 2) + raw_orig.first_time
    durations = np.repeat(1, len(onsets))
    descriptions = ['A', 'B', 'C']
    for onset, duration, description in zip(onsets, durations, descriptions):
        raw_orig.annotations.append(onset, duration, description)
    n_anns = len(raw_orig.annotations)
    fig = raw_orig.plot()
    fig.test_mode = True
    annot_dock = fig.mne.fig_annotation

    # Activate annotation_mode
    fig._fake_keypress('a')

    # Set current description to index 1
    annot_dock.description_cmbx.setCurrentIndex(1)
    assert fig.mne.current_description == 'B'

    # Draw additional annotation
    fig._fake_click((8., 1.), add_points=[(9., 1.)], xform='data', button=1,
                    kind='drag')
    assert len(raw_orig.annotations.onset) == n_anns + 1
    assert len(raw_orig.annotations.duration) == n_anns + 1
    assert len(raw_orig.annotations.description) == n_anns + 1
    assert raw_orig.annotations.description[-1] == 'B'

    # Test remove all regions description
    annot_dock._remove_description('B')
    assert len(raw_orig.annotations.onset) == n_anns - 1
    assert len(raw_orig.annotations.duration) == n_anns - 1
    assert len(raw_orig.annotations.description) == n_anns - 1
    assert fig.mne.current_description == 'A'
    assert fig.mne.selected_region is None

    # Redraw annotation (now with 'A')
    fig._fake_click((4., 1.), add_points=[(5., 1.)], xform='data', button=1,
                    kind='drag')
    assert len(raw_orig.annotations.onset) == n_anns
    assert len(raw_orig.annotations.duration) == n_anns
    assert len(raw_orig.annotations.description) == n_anns

    # Test editing descriptions (all)
    annot_dock._edit_description_all('D')
    assert len(np.where(raw_orig.annotations.description == 'D')[0]) == 2

    # Test editing descriptions (selected)
    # Select second region
    fig._fake_click((4.5, 1.), xform='data')
    assert fig.mne.selected_region.description == 'D'
    annot_dock._edit_description_selected('E')
    assert raw_orig.annotations.description[1] == 'E'

    # Test Spinbox behaviour
    # Update of Spinboxes
    fig._fake_click((2.5, 1.), xform='data')
    assert annot_dock.start_bx.value() == 2.
    assert annot_dock.stop_bx.value() == 3.

    # Setting values with Spinboxex
    annot_dock.start_bx.setValue(1.5)
    annot_dock.start_bx.editingFinished.emit()
    annot_dock.stop_bx.setValue(3.5)
    annot_dock.stop_bx.editingFinished.emit()
    assert raw_orig.annotations.onset[0] == 1.5 + raw_orig.first_time
    assert raw_orig.annotations.duration[0] == 2.

    # Test SpinBox Warning
    annot_dock.start_bx.setValue(6)
    annot_dock.start_bx.editingFinished.emit()
    assert fig.msg_box.isVisible()
    assert fig.msg_box.informativeText() == 'Start can\'t be bigger or ' \
                                            'equal to Stop!'
    fig.msg_box.close()


def test_pg_settings_dialog(raw_orig, pg_backend):
    """Test Settings Dialog toggle on/off for pyqtgraph-backend."""
    fig = raw_orig.plot()
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    QTest.qWait(50)
    assert fig.mne.fig_settings is None
    fig._fake_click_on_toolbar_action('Settings', wait_after=500)
    assert fig.mne.fig_settings is not None
    assert pg_backend._get_n_figs() == 2
    fig._fake_click_on_toolbar_action('Settings', wait_after=500)
    assert pg_backend._get_n_figs() == 1
    assert fig.mne.fig_settings is None
    fig._fake_click_on_toolbar_action('Settings', wait_after=500)
    assert pg_backend._get_n_figs() == 2
    assert fig.mne.fig_settings is not None
    fig._fake_click_on_toolbar_action('Settings', wait_after=500)
    assert pg_backend._get_n_figs() == 1
    assert fig.mne.fig_settings is None

    fig._fake_click_on_toolbar_action('Settings', wait_after=500)
    assert fig.mne.fig_settings is not None
    downsampling_control = fig.mne.fig_settings.downsampling_box
    assert downsampling_control.value() == fig.mne.downsampling

    downsampling_control.setValue(2)
    QTest.qWait(100)
    assert downsampling_control.value() == 2
    assert downsampling_control.value() == fig.mne.downsampling

    downsampling_control.setValue(3)
    QTest.qWait(100)
    assert downsampling_control.value() == 3
    assert downsampling_control.value() == fig.mne.downsampling

    QTest.qWait(100)
    downsampling_method_control = fig.mne.fig_settings.ds_method_cmbx
    assert fig.mne.ds_method == downsampling_method_control.currentText()

    downsampling_method_control.setCurrentText('mean')
    QTest.qWait(100)
    assert downsampling_method_control.currentText() == 'mean'
    assert fig.mne.ds_method == 'mean'

    downsampling_method_control.setCurrentText('subsample')
    QTest.qWait(100)
    assert downsampling_method_control.currentText() == 'subsample'
    assert fig.mne.ds_method == 'subsample'

    downsampling_method_control.setCurrentText('peak')
    QTest.qWait(100)
    assert downsampling_method_control.currentText() == 'peak'
    assert fig.mne.ds_method == 'peak'

    downsampling_method_control.setCurrentText('invalid_method_name')
    QTest.qWait(100)
    assert downsampling_method_control.currentText() != 'invalid_method_name'

    sensitivity_control = fig.mne.fig_settings.scroll_sensitivity_slider
    assert fig.mne.scroll_sensitivity == sensitivity_control.value()

    sensitivity_control.setValue(100)
    QTest.qWait(100)
    assert sensitivity_control.value() == 100
    assert fig.mne.scroll_sensitivity == 100

    QTest.qWait(100)
    sensitivity_values = list(range(sensitivity_control.minimum(),
                                    sensitivity_control.maximum() + 1, 40))
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


def test_pg_help_dialog(raw_orig, pg_backend):
    """Test Settings Dialog toggle on/off for pyqtgraph-backend."""
    fig = raw_orig.plot()
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    QTest.qWait(50)
    assert fig.mne.fig_help is None
    fig._fake_click_on_toolbar_action('Help', wait_after=500)
    assert fig.mne.fig_help is not None
    assert pg_backend._get_n_figs() == 2
    fig._fake_click_on_toolbar_action('Help', wait_after=500)
    assert fig.mne.fig_help is None
    assert pg_backend._get_n_figs() == 1
    fig._fake_click_on_toolbar_action('Help', wait_after=500)
    assert fig.mne.fig_help is not None
    assert pg_backend._get_n_figs() == 2
    fig._fake_click_on_toolbar_action('Help', wait_after=500)
    assert fig.mne.fig_help is None
    assert pg_backend._get_n_figs() == 1


def test_pg_toolbar_time_plus_minus(raw_orig, pg_backend):
    fig = raw_orig.plot()
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    assert pg_backend._get_n_figs() == 1

    min_duration = 3 * np.diff(fig.mne.inst.times[:2])[0]  # hard code.
    xmin, xmax = fig.mne.viewbox.viewRange()[0]
    for _ in range(100):
        if xmax - xmin <= min_duration:
            break
        fig._fake_click_on_toolbar_action('- Time', wait_after=20)
        xmin, xmax = fig.mne.viewbox.viewRange()[0]
    assert xmax - xmin == min_duration

    eps = 0.01
    step = 0.25
    fig._fake_click_on_toolbar_action('+ Time', wait_after=100)
    xmin_new, xmax_new = fig.mne.viewbox.viewRange()[0]
    assert xmax_new - (xmax + (xmax - xmin * step)) < eps

    xmin, xmax = fig.mne.viewbox.viewRange()[0]
    for _ in range(100):
        if xmax + fig.mne.duration * step >= fig.mne.xmax:
            break
        fig._fake_click_on_toolbar_action('+ Time', wait_after=20)
        xmin, xmax = fig.mne.viewbox.viewRange()[0]

    fig._fake_click_on_toolbar_action('+ Time', wait_after=200)
    fig._fake_click_on_toolbar_action('+ Time', wait_after=200)

    xmin, xmax = fig.mne.viewbox.viewRange()[0]
    fig._fake_click_on_toolbar_action('+ Time', wait_after=200)
    xmin_new, xmax_new = fig.mne.viewbox.viewRange()[0]
    assert xmax_new == xmax  # no effect after span maxed

    step = -0.2
    xmin, xmax = fig.mne.viewbox.viewRange()[0]
    fig._fake_click_on_toolbar_action('- Time', wait_after=200)
    xmin_new, xmax_new = fig.mne.viewbox.viewRange()[0]
    assert xmax_new == xmax + ((xmax - xmin) * step)

    xmin, xmax = fig.mne.viewbox.viewRange()[0]
    fig._fake_click_on_toolbar_action('- Time', wait_after=200)
    xmin_new, xmax_new = fig.mne.viewbox.viewRange()[0]
    assert xmax_new == xmax + ((xmax - xmin) * step)

    for _ in range(7):
        fig._fake_click_on_toolbar_action('- Time', wait_after=20)

    assert pg_backend._get_n_figs() == 1  # still alive


def test_pg_toolbar_channels_plus_minus(raw_orig, pg_backend):
    fig = raw_orig.plot()
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    assert pg_backend._get_n_figs() == 1

    if fig.mne.butterfly is not True:
        fig._fake_keypress('b')  # toggle butterfly mode
    fig._fake_click_on_toolbar_action('- Channels', wait_after=100)
    ymin, ymax = fig.mne.viewbox.viewRange()[1]
    fig._fake_click_on_toolbar_action('- Channels', wait_after=100)
    assert [ymin, ymax] == fig.mne.viewbox.viewRange()[1]
    fig._fake_click_on_toolbar_action('+ Channels', wait_after=100)
    assert [ymin, ymax] == fig.mne.viewbox.viewRange()[1]

    if fig.mne.butterfly is True:
        fig._fake_keypress('b')  # toggle butterfly off

    for _ in range(10):
        if ymax - ymin <= 2:
            break
        fig._fake_click_on_toolbar_action('- Channels', wait_after=40)
        ymin, ymax = fig.mne.viewbox.viewRange()[1]
    assert ymax - ymin == 2
    fig._fake_click_on_toolbar_action('- Channels', wait_after=40)
    ymin, ymax = fig.mne.viewbox.viewRange()[1]
    assert ymax - ymin == 2

    step = 10
    fig._fake_click_on_toolbar_action('+ Channels', wait_after=100)
    ymin_new, ymax_new = fig.mne.viewbox.viewRange()[1]
    assert ymax_new == ymax + step

    ymin, ymax = fig.mne.viewbox.viewRange()[1]
    fig._fake_click_on_toolbar_action('+ Channels', wait_after=100)
    ymin_new, ymax_new = fig.mne.viewbox.viewRange()[1]
    assert ymax_new == ymax + step

    ymin, ymax = fig.mne.viewbox.viewRange()[1]
    fig._fake_click_on_toolbar_action('+ Channels', wait_after=100)
    ymin_new, ymax_new = fig.mne.viewbox.viewRange()[1]
    assert ymax_new == ymax + step

    assert pg_backend._get_n_figs() == 1    # still alive


def test_pg_toolbar_zoom(raw_orig, pg_backend):
    fig = raw_orig.plot()
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    assert pg_backend._get_n_figs() == 1

    step = 4 / 5
    scale_factor = fig.mne.scale_factor
    fig._fake_click_on_toolbar_action('Zoom out', wait_after=100)
    scale_factor_new = fig.mne.scale_factor
    assert scale_factor_new == scale_factor * step

    for _ in range(6):
        fig._fake_click_on_toolbar_action('Zoom out', wait_after=100)

    step = 5 / 4
    scale_factor = fig.mne.scale_factor
    fig._fake_click_on_toolbar_action('Zoom in', wait_after=100)
    scale_factor_new = fig.mne.scale_factor
    assert scale_factor_new == scale_factor * step

    for _ in range(6):
        fig._fake_click_on_toolbar_action('Zoom in', wait_after=100)

    assert pg_backend._get_n_figs() == 1  # still alive


def test_pg_toolbar_annotations(raw_orig, pg_backend):
    fig = raw_orig.plot()
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    assert pg_backend._get_n_figs() == 1

    state_annotation_widget = fig.mne.annotation_mode
    fig._fake_click_on_toolbar_action('Annotations', wait_after=100)
    assert fig.mne.annotation_mode != state_annotation_widget

    fig._fake_click_on_toolbar_action('Annotations', wait_after=300)
    fig._fake_click_on_toolbar_action('Annotations', wait_after=300)
    fig._fake_click_on_toolbar_action('Annotations', wait_after=300)

    assert pg_backend._get_n_figs() == 1    # still alive


def test_pg_toolbar_actions(raw_orig, pg_backend):
    """ Test toolbar all actions combined.
        Toolbar actions here create a separate QDialog window.
        We test the state machine for each window toggle button. """
    fig = raw_orig.plot()
    fig.test_mode = True
    QTest.qWaitForWindowExposed(fig)
    assert pg_backend._get_n_figs() == 1

    fig._fake_click_on_toolbar_action('SSP', wait_after=200)
    assert pg_backend._get_n_figs() == 2
    fig._fake_click_on_toolbar_action('Settings', wait_after=200)
    assert pg_backend._get_n_figs() == 3
    fig._fake_click_on_toolbar_action('Settings', wait_after=100)
    assert pg_backend._get_n_figs() == 2
    fig._fake_click_on_toolbar_action('Help', wait_after=200)
    assert pg_backend._get_n_figs() == 3
    fig._fake_click_on_toolbar_action('Settings', wait_after=200)
    assert pg_backend._get_n_figs() == 4
    fig._fake_click_on_toolbar_action('SSP', wait_after=200)
    assert pg_backend._get_n_figs() == 3
    fig._fake_click_on_toolbar_action('Settings', wait_after=100)
    assert pg_backend._get_n_figs() == 2
    fig._fake_click_on_toolbar_action('Help', wait_after=100)
    assert pg_backend._get_n_figs() == 1
