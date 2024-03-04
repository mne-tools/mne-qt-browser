# Contributing to mne-qt-browser

**Thank you very much for taking the time to contribute!** :+1::+1::+1:

As this package is a subsidiarian of [MNE-Python](https://mne.tools/dev/index.html)
its [guidelines](https://github.com/mne-tools/mne-python) for contributing apply as well.


## Setting up development environment

Follow the [instructions from mne-python](https://mne.tools/dev/install/contributing.html#setting-up-your-local-development-environment)
and additionally install the dependencies for this repository:

```
pip install -e ".[tests]"
```

## Modifying icons

The icons are stored in the `icons` folder, which contains two themes "light" and "dark". If you want to modify an existing icon or add a new one, you will need to regenerate the resource file. This can be done by running the following command in the `mne_qt_browser` folder (this requires `PySide6` to be installed):

```
pyside6-rcc icons.qrc -o rc_icons.py
```

The auto-generated `rc_icons.py` will be updated with the new/changed icons. Make sure to include this file in your pull request if you have modified any icons. Please also make sure to apply your changes to both the "light" and "dark" theme icons.
