import mne
import os
import numpy as np

from mne.viz._figure import use_browser_backend


def main():
    sample_data_folder = mne.datasets.sample.data_path()
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
