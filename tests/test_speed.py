import sys
from copy import copy
from functools import partial

import numpy as np
import pytest

bm_limit = 50
bm_count = copy(bm_limit)
hscroll_dir = True
vscroll_dir = True
h_last_time = None
v_last_time = None


# Skip speed-test for CIs
@pytest.mark.skip
@pytest.mark.parametrize('benchmark_param', [{'use_opengl': False},
                                             {'use_opengl': True},
                                             {'precompute': False},
                                             {'precompute': True}])
def test_scroll_speed(raw_orig, benchmark_param, browser_backend):
    """Test the speed of a parameter."""
    # Remove spaces and get params with values
    from time import perf_counter

    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication

    hscroll_diffs = list()
    vscroll_diffs = list()

    def _ininite_hscroll(pg_fig):
        global bm_count
        global hscroll_dir
        global vscroll_dir
        global h_last_time
        global v_last_time

        if bm_count > 0:
            bm_count -= 1
            # Scroll in horizontal direction and turn at ends.
            if pg_fig.mne.t_start + pg_fig.mne.duration \
                    >= pg_fig.mne.inst.times[-1]:
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
                    >= len(pg_fig.mne.inst.ch_names):
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
            bm_count = copy(bm_limit)
            hscroll_dir = True
            vscroll_dir = True
            h_last_time = None
            v_last_time = None

            h_mean_fps = 1 / np.median(hscroll_diffs)
            v_mean_fps = 1 / np.median(vscroll_diffs)
            print(f'The median FPS for {benchmark_param} is:\n'
                  f'Horizontal = {h_mean_fps:.3f} FPS\n'
                  f'Vertical = {v_mean_fps:.3f} FPS')
            pg_fig.close()

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    fig = raw_orig.plot(duration=5, n_channels=40,
                        show=False, block=False, **benchmark_param)
    timer = QTimer()
    timer.timeout.connect(partial(_ininite_hscroll, fig))
    timer.start(0)

    fig.show()
    with pytest.raises(SystemExit):
        sys.exit(app.exec())
