from pathlib import Path

import mne

sample_dir = mne.datasets.sample.data_path()
raw_path = Path(sample_dir) / "MEG" / "sample" / "sample_audvis_raw.fif"
raw = mne.io.read_raw(raw_path)

mne.viz.set_browser_backend("qt")  # Enable mne-qt-browser backend if mne < 1.0
raw.plot(block=True)
