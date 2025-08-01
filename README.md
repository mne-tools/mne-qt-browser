# MNE Qt Browser

![Screenshot of MNE Qt Browser](https://github.com/mne-tools/mne-qt-browser/raw/main/screenshot.png)

MNE Qt Browser is an alternative backend for plotting multichannel time series data (such as EEG or MEG) with [MNE-Python](https://github.com/mne-tools/mne-python). The backend is based on [PyQtGraph](https://github.com/pyqtgraph/pyqtgraph), which in turn uses [Qt](https://www.qt.io/product/framework) under the hood.


## Installation

MNE Qt Browser is not a standalone package—it requires MNE-Python to be installed. The easiest way to use it is by installing MNE-Python through the [official installers](https://mne.tools/stable/install/installers.html), which include the browser by default.

The browser is also supported by [MNELAB](https://github.com/cbrnr/mnelab), a graphical user interface for MNE-Python. The recommended way to install MNELAB is via the official installers as well. In this case, MNE Qt Browser will be installed automatically—you just need to enable it in the settings by selecting *Qt* as the plot backend.

If you already have the `mne` package installed in your Python environment, you can also install `mne-qt-browser` separately (e.g., using `pip`, `uv`, or `conda`).


## Usage

The backend supports plotting for the following MNE-Python methods:

- [`mne.io.Raw.plot()`](https://mne.tools/stable/generated/mne.io.Raw.html)
- [`mne.Epochs.plot()`](https://mne.tools/stable/generated/mne.Epochs.html)
- [`mne.preprocessing.ICA.plot_sources(raw)`](https://mne.tools/stable/generated/mne.preprocessing.ICA.html)
- [`mne.preprocessing.ICA.plot_sources(epochs)`](https://mne.tools/stable/generated/mne.preprocessing.ICA.html)

The following example demonstrates how to read and plot the [MNE sample](https://mne.tools/stable/generated/mne.datasets.sample.data_path.html) dataset:

```python
import mne

raw = mne.io.read_raw(
    mne.datasets.sample.data_path() / "MEG" / "sample" / "sample_audvis_raw.fif"
)

raw.plot(block=True)
```

If the plot does not appear, check the [troubleshooting](#troubleshooting) section below for possible solutions.

MNE ≥ 1.0.0 will automatically use the Qt backend for plotting if it is available. If you want to set the backend explicitly, you can do so by calling:

```python
mne.viz.set_browser_backend("qt")  # or "matplotlib"
```

You can set the backend to `"qt"` or `"matplotlib"`. If you want to make this setting permanent, you can modify your MNE configuration file by running:

```python
mne.set_config("MNE_BROWSER_BACKEND", "qt")  # or "matplotlib"
```


## Troubleshooting

### Running from a script

If you run a script containing `raw.plot()` as follows, the plot will close immediately after the script finishes:

```console
python example_script.py
```

To keep the plot open, you can either use blocking mode:

```python
raw.plot(block=True)
```

Alternatively, you can run the script in interactive mode:

```console
python -i example_script.py
```

### IPython

When using an interactive IPython console, calling `raw.plot()` in non-blocking mode may cause the plot window to freeze or become unresponsive. This happens because IPython must be configured to run the Qt event loop to handle plot interactions.

To fix this, you can either use blocking mode, which runs its own event loop:

```python
raw.plot(block=True)
```

Alternatively, enable Qt event loop integration in your IPython session by running the following magic command before you plot:

```console
%gui qt
```


## Development and testing

You can run the included benchmarks locally with:

```console
pytest -m benchmark mne_qt_browser
```

To run the PyQtGraph tests, use:

```
pytest mne_qt_browser/tests/test_pg_specific.py
```

You can also run additional tests from the MNE-Python repository. The following command assumes that you have cloned the MNE-Python repository in the parent directory of this repository:

```console
pytest -m pgtest ../mne-python/mne/viz/tests
```

These tests require `PyOpenGL` to be installed. If OpenGL is not available on your system, you may encounter errors. To suppress these, add the following line to `mne/conftest.py` *after* the existing `error::` line:

```raw
    ignore:.*PyOpenGL was not found.*:RuntimeWarning
```
