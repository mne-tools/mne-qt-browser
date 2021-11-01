import pathlib

from setuptools import setup

readme = (pathlib.Path(__file__).parent / "README.md").read_text()

setup(name='mne-qt-browser',
      version='0.1.0',
      maintainer='Martin Schulz',
      maintainer_email='dev@earthman-music.de',
      description='A new backend based on pyqtgraph for the 2D-Data-Browser '
                  'in MNE-Python.',
      long_description=readme,
      long_description_content_type='text/markdown',
      license='License :: OSI Approved :: BSD License',
      url='https://github.com/mne-tools/mne-qt-browser',
      download_url='https://github.com/mne-tools/mne-qt-browser/archive/refs'
                   '/tags/v0.1.0.tar.gz',
      project_urls={'Bug Tracker':
                    'https://github.com/mne-tools/mne-qt-browser/issues'},
      classifiers=['Programming Language :: Python :: 3',
                   'License :: OSI Approved :: BSD License',
                   'Operating System :: OS Independent'],
      packages=['mne_qt_browser'],
      install_requires=['numpy',
                        'scipy',
                        'matplotlib',
                        'PyQt5',
                        'qtpy',
                        'mne',
                        'pyqtgraph',
                        'pyopengl'],
      entry_points={'console_scripts':
                       ['mne_qt_browser = mne_qt_browser.__main__:main']}
      )
