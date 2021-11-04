# mne-qt-browser

#### A new backend based on pyqtgraph for the 2D-Data-Browser in MNE-Python.

This repository hosts the code for an alternative backend for plotting 2D-Data with 
[MNE-Python](https://github.com/mne-tools/mne-python).

The backend is based on [pyqtgraph](https://github.com/pyqtgraph/pyqtgraph) 
which uses Qt's [Graphics View Framework](https://doc.qt.io/qt-5/graphicsview.html)
for the plotting.
Development started as a [2021's Google Summer of Code Project](https://github.com/marsipu/gsoc2021).
Currently, only `Raw.plot()` is supported. For the future support for Epochs
and ICA-Sources is planned.

### Usage

Import mne-python
```python
import mne
```
To use the new backend, set `pyqtgraph` as backend with 
```python
mne.viz.set_browser_backend("pyqtgraph")
```
or to set it permanently with
```python
mne.set_config('MNE_BROWSE_BACKEND', 'pyqtgraph')
```

Then load and plot your Raw-data, e.g. by using:
```python
raw = mne.io.read_raw("path to your data")
raw.plot(block=True)
```

If you want to try the browser with the sample-dataset from mne-python, 
run `mne-qt-browser` from the terminal.

### Report Bugs & Feature Requests

Please report bugs and feature requests in the [issues](https://github.com/mne-tools/mne-qt-browser/issues) of this repository.

### Development and testing

You can run a benchmark locally with:

```console
$ pytest -m benchmark mne_qt_browser
```

To run tests, clone mne-python, and then run the PyQtGraph tests with e.g.:
```console
$ pytest -m pgtest ../mne-python/mne/viz/tests
```
If you do not have OpenGL installed, this will currently raise errors, and
you'll need to add this line to `mne/conftest.py` after the `error::` line:
```
    ignore:.*PyOpenGL was not found.*:RuntimeWarning
```
