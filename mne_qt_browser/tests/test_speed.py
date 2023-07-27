# -*- coding: utf-8 -*-
# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Martin Schulz <dev@earthman-music.de>
#
# License: BSD-3-Clause

import sys
from copy import copy
from time import perf_counter

import numpy as np
import pytest
from qtpy.QtCore import QTimer

import mne
from mne_qt_browser.figure import MNEQtBrowser
from mne_qt_browser._pg_figure import _methpartial

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


class _Benchmark:
    def __init__(self, pg_fig, app, store, request):
        self.bm_limit = 50
        self.bm_count = copy(self.bm_limit)
        self.hscroll_dir = True
        self.vscroll_dir = True
        self.h_last_time = None
        self.v_last_time = None
        self.hscroll_diffs = list()
        self.vscroll_diffs = list()
        assert isinstance(pg_fig, MNEQtBrowser)
        assert not pg_fig._closed
        self.pg_fig = pg_fig
        # # Wait max. 10 s for precomputed data to load
        if self.pg_fig.load_thread.isRunning():
            self.pg_fig.load_thread.wait(10000)
        timer = QTimer(self.pg_fig)
        timer.timeout.connect(_methpartial(
            self._initiate_hscroll, store=store, request=request))
        timer.start(0)
        self.pg_fig.show()
        with pytest.raises(SystemExit):
            sys.exit(app.exec())

    def _initiate_hscroll(self, *, store, request):
        if self.bm_count > 0:
            self.bm_count -= 1

            if self.pg_fig.mne.is_epochs:
                t_limit = self.pg_fig.mne.boundary_times[-1]
            else:
                t_limit = self.pg_fig.mne.inst.times[-1]

            # Scroll in horizontal direction and turn at ends.
            if self.pg_fig.mne.t_start + self.pg_fig.mne.duration >= t_limit:
                self.hscroll_dir = False
            elif self.pg_fig.mne.t_start <= 0:
                self.hscroll_dir = True
            key = 'right' if self.hscroll_dir else 'left'
            self.pg_fig._fake_keypress(key)
            # Get time-difference
            now = perf_counter()
            if self.h_last_time is not None:
                self.hscroll_diffs.append(now - self.h_last_time)
            self.h_last_time = now
        elif self.bm_count > -self.bm_limit:
            self.bm_count -= 1
            # Scroll in vertical direction and turn at ends.
            if self.pg_fig.mne.ch_start + self.pg_fig.mne.n_channels \
                    >= len(self.pg_fig.mne.ch_order):
                self.vscroll_dir = False
            elif self.pg_fig.mne.ch_start <= 0:
                self.vscroll_dir = True
            key = 'down' if self.vscroll_dir else 'up'
            self.pg_fig._fake_keypress(key)
            # get time-difference
            now = perf_counter()
            if self.v_last_time is not None:
                self.vscroll_diffs.append(now - self.v_last_time)
            self.v_last_time = now
        else:
            h_mean_fps = 1 / np.median(self.hscroll_diffs)
            v_mean_fps = 1 / np.median(self.vscroll_diffs)
            if self.pg_fig.mne.is_epochs:
                if self.pg_fig.mne.epoch_colors is None:
                    type_key = 'Epochs_unicolor'
                else:
                    type_key = 'Epochs_multicolor'
            else:
                type_key = 'Raw'
            store[type_key][request.node.callspec.id] = dict(
                h=h_mean_fps, v=v_mean_fps)
            self.pg_fig.close()
            del self.pg_fig


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
                          pg_backend, request, qapp):
    """Test the speed of a parameter."""
    # Remove spaces and get params with values
    fig = raw_orig.plot(duration=5, n_channels=40,
                        show=False, block=False, **benchmark_param)
    _Benchmark(fig, qapp, store, request)


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
                                      pg_backend, request, qapp):
    events = np.full((50, 3), [0, 0, 1])
    events[:, 0] = np.arange(0, len(raw_orig), len(raw_orig) / 50) \
        + raw_orig.first_samp
    epochs = mne.Epochs(raw_orig, events, preload=True)
    # Prevent problems with info's locked-stated
    epochs.info._unlocked = True

    fig = epochs.plot(show=False, block=False, events=False, **benchmark_param)
    _Benchmark(fig, qapp, store, request)


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
                                        pg_backend, request, qapp):
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

    fig = epochs.plot(show=False, block=False, events=False, epoch_colors=epoch_colors,
                      **benchmark_param)
    _Benchmark(fig, qapp, store, request)
