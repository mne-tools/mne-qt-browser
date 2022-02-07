# -*- coding: utf-8 -*-
# Author: Martin Schulz <dev@earthman-music.de>
#
# License: BSD-3-Clause

import numpy as np


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
    annot_dock.description_cmbx.activated.emit(1)
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
