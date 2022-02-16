# -*- coding: utf-8 -*-
# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Martin Schulz <dev@earthman-music.de>
#
# License: BSD-3-Clause

import sys
from copy import copy
from functools import partial
from time import perf_counter

import numpy as np
import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

import mne

bm_limit = 50
bm_count = copy(bm_limit)
hscroll_dir = True
vscroll_dir = True
h_last_time = None
v_last_time = None
hscroll_diffs = list()
vscroll_diffs = list()

try:
    import OpenGL  # noqa
except Exception as exc:
    has_gl = False
    reason = str(exc)
else:
    has_gl = True
    reason = ''
gl_mark = pytest.mark.skipif(
        not has_gl, reason=f'Requires PyOpengl (got {reason})')


def _reinit_bm_values():
    global bm_count
    global hscroll_dir
    global vscroll_dir
    global h_last_time
    global v_last_time
    global hscroll_diffs
    global vscroll_diffs

    bm_limit = 50
    bm_count = copy(bm_limit)
    hscroll_dir = True
    vscroll_dir = True
    h_last_time = None
    v_last_time = None
    hscroll_diffs = list()
    vscroll_diffs = list()


def _initiate_hscroll(pg_fig, store, request, timer):
    global bm_count
    global hscroll_dir
    global vscroll_dir
    global h_last_time
    global v_last_time
    if bm_count > 0:
        bm_count -= 1

        if pg_fig.mne.is_epochs:
            t_limit = pg_fig.mne.boundary_times[-1]
        else:
            t_limit = pg_fig.mne.inst.times[-1]

        # Scroll in horizontal direction and turn at ends.
        if pg_fig.mne.t_start + pg_fig.mne.duration >= t_limit:
            hscroll_dir = False
        elif pg_fig.mne.t_start <= 0:
            hscroll_dir = True
        key = 'right' if hscroll_dir else 'left'
        pg_fig._fake_keypress(key)
        # Get time-difference
        now = perf_counter()
        if h_last_time is not None:
            hscroll_diffs.append(now - h_last_time)
        h_last_time = now
    elif bm_count > -bm_limit:
        bm_count -= 1
        # Scroll in vertical direction and turn at ends.
        if pg_fig.mne.ch_start + pg_fig.mne.n_channels \
                >= len(pg_fig.mne.ch_order):
            vscroll_dir = False
        elif pg_fig.mne.ch_start <= 0:
            vscroll_dir = True
        key = 'down' if vscroll_dir else 'up'
        pg_fig._fake_keypress(key)
        # get time-difference
        now = perf_counter()
        if v_last_time is not None:
            vscroll_diffs.append(now - v_last_time)
        v_last_time = now
    else:
        timer.stop()

        h_mean_fps = 1 / np.median(hscroll_diffs)
        v_mean_fps = 1 / np.median(vscroll_diffs)
        if pg_fig.mne.is_epochs:
            if pg_fig.mne.epoch_colors is None:
                type_key = 'Epochs_unicolor'
            else:
                type_key = 'Epochs_multicolor'
        else:
            type_key = 'Raw'
        store[type_key][request.node.callspec.id] = dict(h=h_mean_fps,
                                                         v=v_mean_fps)
        pg_fig.close()


@pytest.mark.benchmark
@pytest.mark.parametrize('benchmark_param', [
    pytest.param({'use_opengl': False, 'precompute': False},
                 id='use_opengl=False'),
    pytest.param({'use_opengl': True, 'precompute': False},
                 id='use_opengl=True', marks=gl_mark),
    pytest.param({'precompute': False, 'use_opengl': False},
                 id='precompute=False'),
    pytest.param({'precompute': True, 'use_opengl': False},
                 id='precompute=True'),
    pytest.param({}, id='defaults'),
])
def test_scroll_speed_raw(raw_orig, benchmark_param, store,
                          pg_backend, request):
    """Test the speed of a parameter."""
    # Remove spaces and get params with values

    _reinit_bm_values()

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    fig = raw_orig.plot(duration=5, n_channels=40,
                        show=False, block=False, **benchmark_param)

    # # Wait max. 10 s for precomputed data to load
    if fig.load_thread.isRunning():
        fig.load_thread.wait(10000)

    timer = QTimer()
    timer.timeout.connect(partial(_initiate_hscroll, fig, store,
                                  request, timer))
    timer.start(0)

    fig.show()
    with pytest.raises(SystemExit):
        sys.exit(app.exec())


def _check_epochs_version():
    import mne
    from packaging.version import parse
    if parse(mne.__version__) <= parse('0.24.1'):
        pytest.skip('Epochs-Test were skipped because of mne <= 0.24.1!')


@pytest.mark.benchmark
@pytest.mark.parametrize('benchmark_param', [
    pytest.param({'use_opengl': False, 'precompute': False},
                 id='use_opengl=False'),
    pytest.param({'use_opengl': True, 'precompute': False},
                 id='use_opengl=True', marks=gl_mark),
    pytest.param({'precompute': False, 'use_opengl': False},
                 id='precompute=False'),
    pytest.param({'precompute': True, 'use_opengl': False},
                 id='precompute=True'),
    pytest.param({}, id='defaults'),
])
def test_scroll_speed_epochs_unicolor(raw_orig, benchmark_param, store,
                                      pg_backend, request):
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication
    _check_epochs_version()
    _reinit_bm_values()

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    events = np.full((50, 3), [0, 0, 1])
    events[:, 0] = np.arange(0, len(raw_orig), len(raw_orig) / 50) \
        + raw_orig.first_samp
    epochs = mne.Epochs(raw_orig, events, preload=True)
    # Prevent problems with info's locked-stated
    epochs.info._unlocked = True

    fig = epochs.plot(show=False, block=False, **benchmark_param)

    # # Wait max. 10 s for precomputed data to load
    if fig.load_thread.isRunning():
        fig.load_thread.wait(10000)

    timer = QTimer()
    timer.timeout.connect(partial(_initiate_hscroll, fig, store,
                                  request, timer))
    timer.start(0)

    fig.show()
    with pytest.raises(SystemExit):
        sys.exit(app.exec())


@pytest.mark.benchmark
@pytest.mark.parametrize('benchmark_param', [
    pytest.param({'use_opengl': False, 'precompute': False},
                 id='use_opengl=False'),
    pytest.param({'use_opengl': True, 'precompute': False},
                 id='use_opengl=True', marks=gl_mark),
    pytest.param({'precompute': False, 'use_opengl': False},
                 id='precompute=False'),
    pytest.param({'precompute': True, 'use_opengl': False},
                 id='precompute=True'),
    pytest.param({}, id='defaults'),
])
def test_scroll_speed_epochs_multicolor(raw_orig, benchmark_param, store,
                                        pg_backend, request):
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication
    _check_epochs_version()
    _reinit_bm_values()

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    events = np.full((50, 3), [0, 0, 1])
    events[:, 0] = np.arange(0, len(raw_orig), len(raw_orig) / 50) \
        + raw_orig.first_samp
    epochs = mne.Epochs(raw_orig, events, preload=True)
    # Prevent problems with info's locked-stated
    epochs.info._unlocked = True
    # Make colored segments (simulating bad epochs,
    # bad segments from autoreject)
    epoch_col1 = np.asarray(['b'] * len(epochs.ch_names))
    epoch_col1[::2] = 'r'
    epoch_col2 = np.asarray(['r'] * len(epochs.ch_names))
    epoch_col2[::2] = 'b'
    epoch_col3 = np.asarray(['g'] * len(epochs.ch_names))
    epoch_col3[::2] = 'b'
    epoch_colors = np.asarray([['b'] * len(epochs.ch_names) for _ in
                               range(len(epochs))])
    epoch_colors[::3] = epoch_col1
    epoch_colors[1::3] = epoch_col2
    epoch_colors[2::3] = epoch_col3
    epoch_colors = epoch_colors.tolist()

    # Multicolored Epochs might be unstable without OpenGL on macOS
    if sys.platform == 'darwin':
        benchmark_param['use_opengl'] = True

    fig = epochs.plot(show=False, block=False, epoch_colors=epoch_colors,
                      **benchmark_param)

    # # Wait max. 10 s for precomputed data to load
    if fig.load_thread.isRunning():
        fig.load_thread.wait(10000)

    timer = QTimer()
    timer.timeout.connect(partial(_initiate_hscroll, fig, store,
                                  request, timer))
    timer.start(0)

    fig.show()
    with pytest.raises(SystemExit):
        sys.exit(app.exec())
