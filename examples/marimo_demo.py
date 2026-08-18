"""Demo of the Qt browser inside marimo: `marimo edit marimo_demo.py`."""

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import mne

    mne.viz.set_browser_backend("qt")
    _path = mne.datasets.sample.data_path()
    raw = mne.io.read_raw_fif(_path / "MEG" / "sample" / "sample_audvis_raw.fif")
    raw.pick("eeg").crop(0, 60).load_data()
    return mo, raw


@app.cell
def _(raw):
    # No setup needed: drag, scroll and click around while marimo waits below.
    fig = raw.plot(duration=10)
    return (fig,)


@app.cell
def _(mo):
    go = mo.ui.run_button(label="Done inspecting -- continue")
    go
    return (go,)


@app.cell
def _(go, mo):
    # This cell won't execute until the button is clicked. Assuming no subsequent cells
    # are running, the window should be interactive. There is no way to gate *all*
    # subsequent cells unless you use `raw.plot(..., block=True)`.
    mo.stop(not go.value)
    print("gate passed; the rest of the notebook can run now")
    return
