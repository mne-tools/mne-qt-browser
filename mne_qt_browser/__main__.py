import os
import sys

import numpy as np

import mne
import mne_qt_browser
from mne.viz._figure import use_browser_backend
from optparse import OptionParser


def main():
    parser = OptionParser(prog='mne-qt-browser',
                          version=mne_qt_browser.__version__,
                          description='Run a demo',
                          epilog=None, usage='usage: %prog [options]')
    options, args = parser.parse_args()  # noqa

    sample_data_folder = mne.datasets.sample.data_path(download=False)
    if sample_data_folder == '':
        print('The sample data must be downloaded with '
              'mne.datasets.sample.data_path() in order to use this function')
        sys.exit(1)
    sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                        'sample_audvis_raw.fif')
    raw = mne.io.read_raw(sample_data_raw_file)

    with use_browser_backend('pyqtgraph'):
        onsets = np.arange(2, 8, 2) + raw.first_time
        durations = np.repeat(1, len(onsets))
        descriptions = ['Test1', 'Test2', 'Test3']
        for onset, duration, description in \
                zip(onsets, durations, descriptions):
            raw.annotations.append(onset, duration, description)

        raw.plot(block=True)


if __name__ == '__main__':
    main()
