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
To use the new backend, set `pyqtgraph` as backend with 
```python
import mne
mne.viz.set_browser_backend("pyqtgraph")
```
or to set it permanently with
```python
import mne
mne.set_config('MNE_BROWSE_BACKEND', 'pyqtgraph')
```

### Report Bugs & Feature Requests
Please report bugs and feature requests in the [issues](https://github.com/mne-tools/mne-qt-browser/issues) of this repository.
