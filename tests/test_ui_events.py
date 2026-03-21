# License: BSD-3-Clause
# Copyright the MNE Qt Browser contributors.

from mne.viz import ui_events
from numpy.testing import assert_allclose


def test_time_change_publishes_on_vline(raw_orig, pg_backend):
    """Test that adding a vline publishes a TimeChange event."""
    raw = raw_orig.copy().crop(tmax=10.0).resample(100)
    fig = raw.plot()
    fig.test_mode = True

    callback_calls = []

    def callback(event):
        callback_calls.append(event)

    ui_events.subscribe(fig, "time_change", callback)

    fig._add_vline(2.0)
    assert len(callback_calls) >= 1
    assert isinstance(callback_calls[-1], ui_events.TimeChange)
    assert_allclose(callback_calls[-1].time, 2.0, atol=0.01)

    fig.close()


def test_time_change_linked_figures(raw_orig, pg_backend):
    """Test that linking two figures syncs vlines via TimeChange."""
    raw = raw_orig.copy().crop(tmax=10.0).resample(100)
    fig1 = raw.plot()
    fig2 = raw.plot()
    fig1.test_mode = True
    fig2.test_mode = True

    ui_events.link(fig1, fig2)

    fig1._add_vline(3.0)
    assert fig2.mne.vline is not None

    fig1.close()
    fig2.close()


def test_time_browse_publishes_on_scroll(raw_orig, pg_backend):
    """Test that scrolling time publishes a TimeBrowse event."""
    raw = raw_orig.copy().crop(tmax=20.0).resample(100)
    fig = raw.plot(duration=5.0)
    fig.test_mode = True

    callback_calls = []

    def callback(event):
        callback_calls.append(event)

    ui_events.subscribe(fig, "time_browse", callback)

    fig.mne.plt.setXRange(2.0, 7.0, padding=0)
    assert len(callback_calls) >= 1
    assert isinstance(callback_calls[-1], ui_events.TimeBrowse)

    fig.close()


def test_time_browse_linked_figures(raw_orig, pg_backend):
    """Test that linking two figures syncs time browsing."""
    raw = raw_orig.copy().crop(tmax=20.0).resample(100)
    fig1 = raw.plot(duration=5.0)
    fig2 = raw.plot(duration=5.0)
    fig1.test_mode = True
    fig2.test_mode = True

    ui_events.link(fig1, fig2)

    fig1.mne.plt.setXRange(3.0, 8.0, padding=0)
    assert_allclose(fig2.mne.t_start, 3.0, atol=0.5)

    fig1.close()
    fig2.close()


def test_channels_select_publishes(raw_orig, pg_backend):
    """Test that scrolling channels publishes a ChannelsSelect event."""
    raw = raw_orig.copy().crop(tmax=10.0).resample(100)
    fig = raw.plot()
    fig.test_mode = True

    callback_calls = []

    def callback(event):
        callback_calls.append(event)

    ui_events.subscribe(fig, "channels_select", callback)

    fig.mne.plt.setYRange(2, 8, padding=0)
    assert len(callback_calls) >= 1
    assert isinstance(callback_calls[-1], ui_events.ChannelsSelect)
    assert hasattr(callback_calls[-1], "ch_names")

    fig.close()


def test_disable_ui_events_prevents_feedback(raw_orig, pg_backend):
    """Test that disable_ui_events prevents event publishing."""
    raw = raw_orig.copy().crop(tmax=10.0).resample(100)
    fig = raw.plot()
    fig.test_mode = True

    callback_calls = []

    def callback(event):
        callback_calls.append(event)

    ui_events.subscribe(fig, "time_change", callback)

    with ui_events.disable_ui_events(fig):
        fig._add_vline(2.0)

    assert len(callback_calls) == 0

    fig.close()


def test_close_unsubscribes(raw_orig, pg_backend):
    """Test that closing figure properly unsubscribes from events."""
    raw = raw_orig.copy().crop(tmax=10.0).resample(100)
    fig = raw.plot()
    fig.test_mode = True

    channel = ui_events._get_event_channel(fig)
    assert "time_change" in channel
    assert "time_browse" in channel
    assert "channels_select" in channel

    fig.close()
    # After close, the event channel should have been cleaned up
    # by the unsubscribe call in closeEvent
    channel = ui_events._get_event_channel(fig)
    assert "time_change" not in channel
    assert "time_browse" not in channel
    assert "channels_select" not in channel
