# mne-qt-browser
#### A new backend based on pyqtgraph for the 2D-Data-Browser in MNE-Python.

This repository hosts the code for an alternative backend for plotting 2D-Data with 
[MNE-Python](https://github.com/mne-tools/mne-python) (e.g. with `Raw.plot()`).

The backend is based on [pyqtgraph](https://github.com/pyqtgraph/pyqtgraph) 
which uses Qt's [Graphics View Framework](https://doc.qt.io/qt-5/graphicsview.html)
for the plotting.
Development started as a [2021's Google Summer of Code Project](https://github.com/marsipu/gsoc2021).

### Installation
_Currently, the changes to integrate the new backend are still in this [PR](https://github.com/mne-tools/mne-python/pull/9687). 
You need to fork it to run the new backend._

To use the new backend, set `pyqtgraph` as backend with 
```
mne.viz.set_browse_backend("pyqtgraph")
``` 

You will be prompted to install `mne_qt_browser` if you haven't installed it yet.
If installation should fail, you can install manually with:
```
pip install https://github.com/mne-tools/mne-qt-browser/zipball/main
```

### Report Bugs & Feature Requests
Please report bugs and feature requests in the [issues](https://github.com/mne-tools/mne-qt-browser/issues) of this repository.

### Current Project
I set up a [project](https://github.com/mne-tools/mne-qt-browser/projects/1) with ToDo's for the upcoming MNE-Python Release 0.24.
