# License: BSD-3-Clause
# Copyright the MNE Qt Browser contributors.

"""Base classes and functions for 2D browser backends."""

import gc
import inspect
import math
import platform
import sys
import weakref
from ast import literal_eval
from os.path import getsize
from pathlib import Path

import numpy as np

try:
    from qtpy.QtCore import Qt
except Exception as e:
    if e.__class__.__name__ == "QtBindingsNotFoundError":
        raise ImportError(
            "No Qt binding found, please install PySide6 or PyQt6."
        ) from None
    else:
        raise

import scooby
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import to_rgba_array
from mne.annotations import _sync_onset
from mne.utils import _check_option, get_config, logger, sizeof_fmt, warn
from mne.viz._figure import BrowserBase
from mne.viz.backends._utils import _init_mne_qtapp, _qt_raise_window
from mne.viz.utils import _merge_annotations, _simplify_float
from pyqtgraph import (
    InfiniteLine,
    PlotItem,
    Point,
    mkPen,
    setConfigOption,
)
from qtpy.QtCore import (
    QEvent,
    QSettings,
    QSignalBlocker,
    QThread,
    Signal,
)
from qtpy.QtGui import QIcon, QMouseEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import (
    QAction,
    QActionGroup,
    QApplication,
    QGraphicsView,
    QGridLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QRadioButton,
    QToolButton,
    QWidget,
    QWidgetAction,
)
from scipy.stats import zscore

from mne_qt_browser import _browser_instances
from mne_qt_browser._colors import _get_color, _rgb_to_lab
from mne_qt_browser._dialogs import (
    HelpDialog,
    ProjDialog,
    SelectionDialog,
    SettingsDialog,
    _BaseDialog,
)
from mne_qt_browser._fixes import capture_exceptions
from mne_qt_browser._graphic_items import (
    AnnotRegion,
    Crosshair,
    DataTrace,
    EventLine,
    ScaleBar,
    ScaleBarText,
    VLine,
)
from mne_qt_browser._utils import (
    DATA_CH_TYPES_ORDER,
    _disconnect,
    _get_channel_scaling,
    _methpartial,
    _safe_splash,
    _screen_geometry,
    _set_window_flags,
    qsettings_params,
)
from mne_qt_browser._widgets import (
    AnnotationDock,
    BrowserView,
    ChannelAxis,
    ChannelScrollBar,
    OverviewBar,
    RawViewBox,
    TimeAxis,
    TimeScrollBar,
)

name = "pyqtgraph"


class LoadThread(QThread):
    """A worker object for precomputing in a separate QThread."""

    loadProgress = Signal(int)
    processText = Signal(str)
    loadingFinished = Signal()

    def __init__(self, browser):
        super().__init__()
        self.weakbrowser = weakref.ref(browser)
        self.mne = browser.mne
        self.loadProgress.connect(self.mne.load_progressbar.setValue)
        self.processText.connect(_methpartial(browser._show_process))
        self.loadingFinished.connect(_methpartial(browser._precompute_finished))

    def run(self):
        """Load and process data in a separate QThread."""
        # Split data loading into 10 chunks to show user progress.
        # Testing showed that e.g. n_chunks=100 extends loading time
        # (at least for the sample dataset)
        # because of the frequent gui-update-calls.
        # Thus n_chunks = 10 should suffice.
        data = None
        if self.mne.is_epochs:
            times = (
                np.arange(len(self.mne.inst) * len(self.mne.inst.times))
                / self.mne.info["sfreq"]
            )
        else:
            times = None
        n_chunks = min(10, len(self.mne.inst))
        chunk_size = len(self.mne.inst) // n_chunks
        browser = self.weakbrowser()
        for n in range(n_chunks):
            start = n * chunk_size
            if n == n_chunks - 1:
                # Get last chunk which may be larger due to rounding above
                stop = None
            else:
                stop = start + chunk_size
            # Load epochs
            if self.mne.is_epochs:
                kwargs = dict()
                if "copy" in inspect.getfullargspec(self.mne.inst.get_data).kwonlyargs:
                    kwargs["copy"] = False
                item = slice(start, stop)
                with self.mne.inst.info._unlock():
                    data_chunk = np.concatenate(
                        self.mne.inst.get_data(item=item, **kwargs), axis=-1
                    )
            # Load raw
            else:
                data_chunk, times_chunk = browser._load_data(start, stop)
                if times is None:
                    times = times_chunk
                else:
                    times = np.concatenate((times, times_chunk), axis=0)

            if data is None:
                data = data_chunk
            else:
                data = np.concatenate((data, data_chunk), axis=1)

            self.loadProgress.emit(n + 1)

        picks = self.mne.ch_order
        # Deactive remove dc because it will be removed for visible range
        stashed_remove_dc = self.mne.remove_dc
        self.mne.remove_dc = False
        data = browser._process_data(data, 0, data.shape[-1], picks, self)
        self.mne.remove_dc = stashed_remove_dc

        ch_type_ordered = self.mne.ch_types[self.mne.ch_order]
        for chii in range(data.shape[0]):
            ch_type = ch_type_ordered[chii]
            data[chii, :] *= self.mne.scalings[ch_type]

        self.mne.global_data = data
        self.mne.global_times = times

        # Calculate Z-Scores
        self.processText.emit("Calculating Z-Scores...")
        browser._get_zscore(data)
        del browser

        self.loadingFinished.emit()

    def clean(self):  # noqa: D102
        if self.isRunning():
            wait_time = 10  # max. waiting time in seconds
            logger.info(
                f"Waiting for Loading-Thread to finish... (max. {wait_time} sec)"
            )
            self.wait(int(wait_time * 1e3))
        _disconnect(self.loadProgress)
        _disconnect(self.processText)
        _disconnect(self.loadingFinished)
        del self.mne
        del self.weakbrowser


class _PGMetaClass(type(QMainWindow), type(BrowserBase)):
    """Class is necessary to prevent a metaclass conflict.

    The conflict arises due to the different types of QMainWindow and
    BrowserBase.
    """

    pass


class MNEQtBrowser(BrowserBase, QMainWindow, metaclass=_PGMetaClass):
    """A PyQtGraph-backend for 2D data browsing."""

    gotClosed = Signal()

    @_safe_splash
    def __init__(self, **kwargs):
        self.backend_name = "pyqtgraph"
        self._closed = False

        BrowserBase.__init__(self, **kwargs)
        QMainWindow.__init__(self)

        # Add to list to keep a reference and avoid premature
        # garbage-collection.
        _browser_instances.append(self)

        # Set the browser style
        try:
            from mne.viz.backends._utils import _qt_get_stylesheet
        except Exception:
            stylesheet = None
        else:
            stylesheet = _qt_get_stylesheet(getattr(self.mne, "theme", "auto"))
        if stylesheet is not None:
            self.setStyleSheet(stylesheet)

        if self.mne.window_title is not None:
            self.setWindowTitle(self.mne.window_title)
        QApplication.processEvents()  # needs to happen for the theme to be set

        # HiDPI stuff
        self._pixel_ratio = self.devicePixelRatio()
        logger.debug(f"Desktop pixel ratio: {self._pixel_ratio:0.3f}")
        self.mne.mkPen = _methpartial(self._hidpi_mkPen)

        bgcolor = self.palette().color(self.backgroundRole()).getRgbF()[:3]
        self.mne.dark = _rgb_to_lab(bgcolor)[0] < 50

        # Prepend our icon search path and set fallback name
        icons_path = f"{Path(__file__).parent}/icons"
        QIcon.setThemeSearchPaths([icons_path] + QIcon.themeSearchPaths())
        QIcon.setFallbackThemeName("light")

        # control raising with _qt_raise_window
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)

        # Initialize attributes which are only used by pyqtgraph, not by
        # matplotlib and add them to MNEBrowseParams.

        # Exactly one MessageBox for messages to facilitate testing/debugging
        self.msg_box = QMessageBox(self)
        # MessageBox modality needs to be adapted for tests
        # (otherwise test execution blocks)
        self.test_mode = False
        # A Settings-Dialog
        self.mne.fig_settings = None
        # Stores decimated data
        self.mne.decim_data = None
        # Stores ypos for selection-mode
        self.mne.selection_ypos_dict = dict()
        # Parameters for precomputing
        self.mne.enable_precompute = False
        self.mne.data_precomputed = False
        self._rerun_load_thread = False
        self.mne.zscore_rgba = None
        # Container for traces
        self.mne.traces = list()
        # Scale-Factor
        self.mne.scale_factor = 1
        # DPI
        screen = QApplication.primaryScreen()
        self.mne.dpi = screen.physicalDotsPerInch()

        # Aspect Ratio
        self.mne.aspect_ratio = screen.geometry().width() / screen.geometry().height()
        # Stores channel-types for butterfly-mode
        self.mne.butterfly_type_order = [
            tp for tp in DATA_CH_TYPES_ORDER if tp in self.mne.ch_types
        ]
        if self.mne.is_epochs:
            # Stores parameters for epochs
            self.mne.epoch_dur = np.diff(self.mne.boundary_times[:2])[0]
            epoch_idx = np.searchsorted(
                self.mne.midpoints,
                (self.mne.t_start, self.mne.t_start + self.mne.duration),
            )
            self.mne.epoch_idx = np.arange(epoch_idx[0], epoch_idx[1])

        # Load from QSettings if available
        for qparam in qsettings_params:
            default = qsettings_params[qparam]
            qvalue = QSettings("mne-tools", "mne-qt-browser").value(
                qparam, defaultValue=default
            )
            # QSettings may alter types depending on OS
            if not isinstance(qvalue, type(default)):
                try:
                    qvalue = literal_eval(qvalue)
                except (SyntaxError, ValueError):
                    if qvalue in ["true", "false"]:
                        qvalue = bool(qvalue)
                    else:
                        qvalue = default
            setattr(self.mne, qparam, qvalue)

        # Initialize channel-colors for faster indexing later
        self.mne.ch_color_ref = dict()
        for idx, ch_name in enumerate(self.mne.ch_names):
            ch_type = self.mne.ch_types[idx]
            self.mne.ch_color_ref[ch_name] = self.mne.ch_color_dict[ch_type]

        # Initialize epoch colors for faster indexing later
        if self.mne.is_epochs:
            if self.mne.epoch_colors is None:
                self.mne.epoch_color_ref = np.repeat(
                    [to_rgba_array(c) for c in self.mne.ch_color_ref.values()],
                    len(self.mne.inst),
                    axis=1,
                )
            else:
                self.mne.epoch_color_ref = np.empty(
                    (len(self.mne.ch_names), len(self.mne.inst), 4)
                )
                for epo_idx, epo in enumerate(self.mne.epoch_colors):
                    for ch_idx, color in enumerate(epo):
                        self.mne.epoch_color_ref[ch_idx, epo_idx] = to_rgba_array(color)

            # Mark bad epochs
            self.mne.epoch_color_ref[:, self.mne.bad_epochs] = to_rgba_array(
                self.mne.epoch_color_bad
            )

            # Mark bad channels
            bad_idxs = np.isin(self.mne.ch_names, self.mne.info["bads"])
            self.mne.epoch_color_ref[bad_idxs, :] = to_rgba_array(self.mne.ch_color_bad)

        # Add Load-Progressbar for loading in a thread
        self.mne.load_prog_label = QLabel("Loading...")
        self.statusBar().addWidget(self.mne.load_prog_label)
        self.mne.load_prog_label.hide()
        self.mne.load_progressbar = QProgressBar()
        # Set to n_chunks of LoadRunner
        self.mne.load_progressbar.setMaximum(10)
        self.statusBar().addWidget(self.mne.load_progressbar, stretch=1)
        self.mne.load_progressbar.hide()

        # A QThread for preloading
        self.load_thread = LoadThread(self)

        # Create centralWidget and layout
        widget = QWidget()
        layout = QGridLayout()

        # Initialize Axis-Items
        self.mne.time_axis = TimeAxis(self.mne)
        if self.mne.is_epochs:
            self.mne.time_axis.setLabel(text="Epoch Index", units=None)
        else:
            self.mne.time_axis.setLabel(text="Time", units="s")

        self.mne.channel_axis = ChannelAxis(self)
        self.mne.viewbox = RawViewBox(self)

        # Start precomputing if enabled
        self._init_precompute()

        # Parameters for overviewbar
        self.mne.overview_mode = getattr(self.mne, "overview_mode", "channels")
        overview_items = dict(
            empty="Empty",
            channels="Channels",
        )
        if self.mne.enable_precompute:
            overview_items["zscore"] = "Z-Score"
        elif self.mne.overview_mode == "zscore":
            warn(
                "Cannot use z-score mode without precomputation, setting "
                'overview_mode="channels"'
            )
            self.mne.overview_mode = "channels"
        _check_option(
            "overview_mode", self.mne.overview_mode, list(overview_items) + ["hidden"]
        )
        if self.mne.overview_mode == "hidden":
            self.mne.overview_visible = False
            self.mne.overview_mode = "channels"

        # Initialize data (needed in DataTrace.update_data).
        self._update_data()

        # Initialize Trace-Plot
        self.mne.plt = PlotItem(
            viewBox=self.mne.viewbox,
            axisItems={"bottom": self.mne.time_axis, "left": self.mne.channel_axis},
        )
        # Hide AutoRange-Button
        self.mne.plt.hideButtons()
        # Configure XY-Range
        if self.mne.is_epochs:
            self.mne.xmax = (
                len(self.mne.inst.times) * len(self.mne.inst) / self.mne.info["sfreq"]
            )
        else:
            self.mne.xmax = self.mne.inst.times[-1]
        # Add one empty line as padding at top (y=0).
        # Negative Y-Axis to display channels from top.
        self.mne.ymax = len(self.mne.ch_order) + 1
        self.mne.plt.setLimits(xMin=0, xMax=self.mne.xmax, yMin=0, yMax=self.mne.ymax)
        # Connect Signals from PlotItem
        self.mne.plt.sigXRangeChanged.connect(self._xrange_changed)
        self.mne.plt.sigYRangeChanged.connect(self._yrange_changed)

        # Add traces
        for ch_idx in self.mne.picks:
            DataTrace(self, ch_idx)

        # Initialize Epochs Grid
        if self.mne.is_epochs:
            grid_pen = self.mne.mkPen(color="k", width=2, style=Qt.DashLine)
            for x_grid in self.mne.boundary_times[1:-1]:
                grid_line = InfiniteLine(pos=x_grid, pen=grid_pen, movable=False)
                self.mne.plt.addItem(grid_line)

        # Add events
        if getattr(self.mne, "event_nums", None) is not None:
            self.mne.events_visible = True
            for ev_time, ev_id in zip(self.mne.event_times, self.mne.event_nums):
                color = self.mne.event_color_dict[ev_id]
                label = self.mne.event_id_rev.get(ev_id, ev_id)
                event_line = EventLine(self.mne, ev_time, label, color)
                self.mne.event_lines.append(event_line)
        else:
            self.mne.events_visible = False

        # Add Scale-Bars
        self._add_scalebars()

        # Check for OpenGL
        # If a user doesn't specify whether or not to use it:
        # 1. If on macOS, enable it by default to avoid segfault
        # 2. Otherwise, disable it (performance differences seem minimal, and
        #    PyOpenGL is an optional requirement)
        opengl_key = "MNE_BROWSER_USE_OPENGL"
        if self.mne.use_opengl is None:  # default: opt-in
            # OpenGL needs to be enabled on macOS
            # (https://github.com/mne-tools/mne-qt-browser/issues/53)
            default = "true" if platform.system() == "Darwin" else ""
            config_val = get_config(opengl_key, default).lower()
            self.mne.use_opengl = config_val == "true"

        if self.mne.use_opengl:
            try:
                import OpenGL
            except (ModuleNotFoundError, ImportError) as exc:
                # On macOS, if use_opengl is True we raise an error because
                # it can lead to segfaults. If a user really knows what they
                # are doing, they can pass use_opengl=False (or set
                # MNE_BROWSER_USE_OPENGL=false)
                if platform.system() == "Darwin":
                    raise RuntimeError(
                        "Plotting on macOS without OpenGL may be unstable! "
                        "We recommend installing PyOpenGL, but it could not "
                        f"be imported, got:\n{exc}\n\n"
                        "If you want to try plotting without OpenGL, "
                        "you can pass use_opengl=False (use at your own "
                        "risk!). If you know non-OpenGL plotting is stable "
                        "on your system, you can also set the config value "
                        f"{opengl_key}=false to permanently change "
                        "the default behavior on your system."
                    ) from None
                # otherwise, emit a warning
                warn(
                    "PyOpenGL was not found and OpenGL cannot be used. "
                    "Consider installing pyopengl with pip or conda or set "
                    '"use_opengl=False" to avoid this warning.'
                )
                self.mne.use_opengl = False
            else:
                logger.info(f"Using pyopengl with version {OpenGL.__version__}")
        # Initialize BrowserView (inherits QGraphicsView)
        self.mne.view = BrowserView(
            self.mne.plt, useOpenGL=self.mne.use_opengl, background="w"
        )
        bgcolor = getattr(self.mne, "bgcolor", "w")
        self.mne.view.setBackground(_get_color(bgcolor, self.mne.dark))
        layout.addWidget(self.mne.view, 0, 0)

        # Initialize Scroll-Bars
        self.mne.ax_hscroll = TimeScrollBar(self.mne)
        layout.addWidget(self.mne.ax_hscroll, 1, 0, 1, 2)

        self.mne.ax_vscroll = ChannelScrollBar(self.mne)
        layout.addWidget(self.mne.ax_vscroll, 0, 1)

        # Initialize VLine
        self.mne.vline = None
        self.mne.vline_visible = False

        # Initialize crosshair (as in pyqtgraph example)
        self.mne.crosshair_enabled = False
        self.mne.crosshair_h = None
        self.mne.crosshair = None
        self.mne.view.sigSceneMouseMoved.connect(self._mouse_moved)

        # Initialize Annotation-Widgets
        self.mne.annotation_mode = False
        if not self.mne.is_epochs:
            self._init_annot_mode()

        # OverviewBar
        self.mne.overview_bar = OverviewBar(self)
        layout.addWidget(self.mne.overview_bar, 2, 0, 1, 2)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Initialize Selection-Dialog
        if getattr(self.mne, "group_by", None) in ["position", "selection"]:
            self._create_selection_fig()

        # Initialize Projectors-Dialog if show_options=True
        if getattr(self.mne, "show_options", False):
            self._toggle_proj_fig()

        # Initialize Toolbar
        self.mne.toolbar = self.addToolBar("Tools")
        # tool_button_style = Qt.ToolButtonTextBesideIcon
        tool_button_style = Qt.ToolButtonIconOnly
        self.mne.toolbar.setToolButtonStyle(tool_button_style)

        adecr_time = QAction(
            icon=self._qicon("less_time"),
            text="Show fewer time points",
            parent=self,
        )
        adecr_time.triggered.connect(_methpartial(self.change_duration, step=-0.2))
        self.mne.toolbar.addAction(adecr_time)
        aincr_time = QAction(
            icon=self._qicon("more_time"), text="Show more time points", parent=self
        )
        aincr_time.triggered.connect(_methpartial(self.change_duration, step=0.25))
        self.mne.toolbar.addAction(aincr_time)
        self.mne.toolbar.addSeparator()

        adecr_nchan = QAction(
            icon=self._qicon("less_channels"),
            text="Show fewer channels",
            parent=self,
        )
        adecr_nchan.triggered.connect(_methpartial(self.change_nchan, step=-10))
        self.mne.toolbar.addAction(adecr_nchan)
        aincr_nchan = QAction(
            icon=self._qicon("more_channels"),
            text="Show more channels",
            parent=self,
        )
        aincr_nchan.triggered.connect(_methpartial(self.change_nchan, step=10))
        self.mne.toolbar.addAction(aincr_nchan)
        self.mne.toolbar.addSeparator()

        adecr_nchan = QAction(
            icon=self._qicon("zoom_out"), text="Reduce amplitude", parent=self
        )
        adecr_nchan.triggered.connect(_methpartial(self.scale_all, step=4 / 5))
        self.mne.toolbar.addAction(adecr_nchan)
        aincr_nchan = QAction(
            icon=self._qicon("zoom_in"), text="Increase amplitude", parent=self
        )
        aincr_nchan.triggered.connect(_methpartial(self.scale_all, step=5 / 4))
        self.mne.toolbar.addAction(aincr_nchan)
        self.mne.toolbar.addSeparator()

        if not self.mne.is_epochs:
            atoggle_annot = QAction(
                icon=self._qicon("annotations"),
                text="Toggle annotations mode",
                parent=self,
            )
            atoggle_annot.triggered.connect(self._toggle_annotation_fig)
            self.mne.toolbar.addAction(atoggle_annot)

        atoggle_proj = QAction(
            icon=self._qicon("ssp"), text="Show projectors", parent=self
        )
        atoggle_proj.triggered.connect(self._toggle_proj_fig)
        self.mne.toolbar.addAction(atoggle_proj)

        atoggle_crosshair = QAction(
            self._qicon("crosshair"), "Toggle crosshair", parent=self
        )
        atoggle_crosshair.setCheckable(True)
        atoggle_crosshair.triggered.connect(self._toggle_crosshair)
        self.mne.crosshair_action = atoggle_crosshair
        self.mne.toolbar.addAction(atoggle_crosshair)

        button = QToolButton(self.mne.toolbar)
        button.setToolTip(
            "<h2>Overview Modes</h2>"
            "<ul>"
            "<li>Empty:<br>"
            "Display no background.</li>"
            "<li>Channels:<br>"
            "Display each channel with its channel type color.</li>"
            "<li>Z-score:<br>"
            "Display the z-score for the data from each channel across time. "
            "Red indicates high z-scores, blue indicates low z-scores, "
            "and the boundaries of the color gradient are defined by the "
            "minimum and maximum z-scores."
            'This only works if precompute is set to "True", or if it is '
            'enabled with "auto" and enough free RAM is available.</li>'
            "</ul>"
        )
        button.setText("Overview Bar")
        button.setIcon(self._qicon("overview_bar"))
        button.setToolButtonStyle(tool_button_style)
        menu = self.mne.overview_menu = QMenu(button)
        group = QActionGroup(menu)
        for key, text in overview_items.items():
            radio = QRadioButton(menu)
            radio.setText(text)
            if key == self.mne.overview_mode:
                radio.setChecked(True)
            action = QWidgetAction(menu)
            action.setDefaultWidget(radio)
            menu.addAction(action)
            group.addAction(action)
            radio.clicked.connect(
                _methpartial(self._overview_radio_clicked, menu=menu, new_mode=key)
            )
        menu.addSeparator()
        visible = QAction("Visible", parent=menu)
        menu.addAction(visible)
        visible.setCheckable(True)
        visible.setChecked(self.mne.overview_visible)
        self.mne.overview_bar.setVisible(self.mne.overview_visible)
        visible.triggered.connect(self._toggle_overview_bar)

        button.setMenu(self.mne.overview_menu)
        button.setPopupMode(QToolButton.InstantPopup)
        self.mne.toolbar.addWidget(button)

        self.mne.toolbar.addSeparator()

        asettings = QAction(self._qicon("settings"), "Settings", parent=self)
        asettings.triggered.connect(self._toggle_settings_fig)
        self.mne.toolbar.addAction(asettings)

        ahelp = QAction(self._qicon("help"), "Help", parent=self)
        ahelp.triggered.connect(self._toggle_help_fig)
        self.mne.toolbar.addAction(ahelp)

        # Set Start-Range (after all necessary elements are initialized)
        self.mne.plt.setXRange(
            self.mne.t_start, self.mne.t_start + self.mne.duration, padding=0
        )
        if self.mne.butterfly:
            self._set_butterfly(True)
        else:
            self.mne.plt.setYRange(0, self.mne.n_channels + 1, padding=0)

        # Set Size
        width = int(self.mne.figsize[0] * self.logicalDpiX())
        height = int(self.mne.figsize[1] * self.logicalDpiY())
        self.resize(width, height)

        # Initialize Keyboard-Shortcuts
        is_mac = platform.system() == "Darwin"
        dur_keys = ("fn + ←", "fn + →") if is_mac else ("Home", "End")
        ch_keys = ("fn + ↑", "fn + ↓") if is_mac else ("Page up", "Page down")
        hscroll_type = "1 epoch" if self.mne.is_epochs else "¼ page"

        self.mne.keyboard_shortcuts = {
            "left": {
                "alias": "←",
                "qt_key": Qt.Key_Left,
                "modifier": [None, "Shift"],
                "slot": [self.hscroll],
                "parameter": ["left", "-full"],
                "description": [
                    f"Scroll left ({hscroll_type})",
                    "Scroll left (full page)",
                ],
            },
            "right": {
                "alias": "→",
                "qt_key": Qt.Key_Right,
                "modifier": [None, "Shift"],
                "slot": [self.hscroll],
                "parameter": ["right", "+full"],
                "description": [
                    f"Scroll right ({hscroll_type})",
                    "Scroll right (full page)",
                ],
            },
            "up": {
                "alias": "↑",
                "qt_key": Qt.Key_Up,
                "slot": [self.vscroll],
                "parameter": ["-full"],
                "description": ["Scroll up (full page)"],
            },
            "down": {
                "alias": "↓",
                "qt_key": Qt.Key_Down,
                "slot": [self.vscroll],
                "parameter": ["+full"],
                "description": ["Scroll down (full page)"],
            },
            "home": {
                "alias": dur_keys[0],
                "qt_key": Qt.Key_Home,
                "kw": "step",
                "slot": [self.change_duration],
                "parameter": [-0.2],
                "description": [f"Decrease duration ({hscroll_type})"],
            },
            "end": {
                "alias": dur_keys[1],
                "qt_key": Qt.Key_End,
                "kw": "step",
                "slot": [self.change_duration],
                "parameter": [0.25],
                "description": [f"Increase duration ({hscroll_type})"],
            },
            "pagedown": {
                "alias": ch_keys[0],
                "qt_key": Qt.Key_PageDown,
                "modifier": [None, "Shift"],
                "kw": "step",
                "slot": [self.change_nchan],
                "parameter": [-1, -10],
                "description": [
                    "Decrease shown channels (1)",
                    "Decrease shown channels (10)",
                ],
            },
            "pageup": {
                "alias": ch_keys[1],
                "qt_key": Qt.Key_PageUp,
                "modifier": [None, "Shift"],
                "kw": "step",
                "slot": [self.change_nchan],
                "parameter": [1, 10],
                "description": [
                    "Increase shown channels (1)",
                    "Increase shown channels (10)",
                ],
            },
            "-": {
                "qt_key": Qt.Key_Minus,
                "slot": [self.scale_all],
                "kw": "step",
                "parameter": [4 / 5],
                "description": ["Decrease Scale"],
            },
            "+": {
                "qt_key": Qt.Key_Plus,
                "slot": [self.scale_all],
                "kw": "step",
                "parameter": [5 / 4],
                "description": ["Increase Scale"],
            },
            "=": {
                "qt_key": Qt.Key_Equal,
                "slot": [self.scale_all],
                "kw": "step",
                "parameter": [5 / 4],
                "description": ["Increase Scale"],
            },
            "a": {
                "qt_key": Qt.Key_A,
                "slot": [self._toggle_annotation_fig, self._toggle_annotations],
                "modifier": [None, "Shift"],
                "description": ["Toggle Annotation Tool", "Toggle Annotations visible"],
            },
            "b": {
                "qt_key": Qt.Key_B,
                "slot": [self._toggle_butterfly],
                "description": ["Toggle Butterfly"],
            },
            "d": {
                "qt_key": Qt.Key_D,
                "slot": [self._toggle_dc],
                "description": ["Toggle DC Correction"],
            },
            "e": {
                "qt_key": Qt.Key_E,
                "slot": [self._toggle_events],
                "description": ["Toggle Events visible"],
            },
            "h": {
                "qt_key": Qt.Key_H,
                "slot": [self._toggle_epoch_histogram],
                "description": ["Toggle Epochs Histogram"],
            },
            "j": {
                "qt_key": Qt.Key_J,
                "slot": [self._toggle_proj_fig, self._toggle_all_projs],
                "modifier": [None, "Shift"],
                "description": ["Toggle Projection Figure", "Toggle all projections"],
            },
            "l": {
                "qt_key": Qt.Key_L,
                "slot": [self._toggle_antialiasing],
                "description": ["Toggle Antialiasing"],
            },
            "o": {
                "qt_key": Qt.Key_O,
                "slot": [self._toggle_overview_bar],
                "description": ["Toggle Overview Bar"],
            },
            "t": {
                "qt_key": Qt.Key_T,
                "slot": [self._toggle_time_format],
                "description": ["Toggle Time Format"],
            },
            "s": {
                "qt_key": Qt.Key_S,
                "slot": [self._toggle_scalebars],
                "description": ["Toggle Scalebars"],
            },
            "w": {
                "qt_key": Qt.Key_W,
                "slot": [self._toggle_whitening],
                "description": ["Toggle Whitening"],
            },
            "x": {
                "qt_key": Qt.Key_X,
                "slot": [self._toggle_crosshair],
                "description": ["Toggle Crosshair"],
            },
            "z": {
                "qt_key": Qt.Key_Z,
                "slot": [self._toggle_zenmode],
                "description": ["Toggle Zen Mode"],
            },
            "?": {
                "qt_key": Qt.Key_Question,
                "slot": [self._toggle_help_fig],
                "description": ["Show Help"],
            },
            "f11": {
                "qt_key": Qt.Key_F11,
                "slot": [self._toggle_fullscreen],
                "description": ["Toggle Fullscreen"],
            },
            "escape": {
                "qt_key": Qt.Key_Escape,
                "slot": [self._check_close],
                "description": ["Close"],
            },
            # Just for testing
            "enter": {"qt_key": Qt.Key_Enter},
            " ": {"qt_key": Qt.Key_Space},
        }
        if self.mne.is_epochs:
            # Disable time format toggling
            del self.mne.keyboard_shortcuts["t"]
        else:
            if self.mne.info["meas_date"] is None:
                del self.mne.keyboard_shortcuts["t"]
            # disable histogram of epoch PTP amplitude
            del self.mne.keyboard_shortcuts["h"]

    def _save_setting(self, key, value):
        """Save a setting to QSettings."""
        QSettings("mne-tools", "mne-qt-browser").setValue(key, value)

    def _hidpi_mkPen(self, *args, **kwargs):
        kwargs["width"] = self._pixel_ratio * kwargs.get("width", 1.0)
        return mkPen(*args, **kwargs)

    def _update_yaxis_labels(self):
        self.mne.channel_axis.repaint()

    def _add_scalebars(self):
        """Add scalebars for all channel-types.

        (scene handles showing them in when in view
        range)
        """
        self.mne.scalebars.clear()
        # To keep order (np.unique sorts)
        ordered_types = self.mne.ch_types[self.mne.ch_order]
        unique_type_idxs = np.unique(ordered_types, return_index=True)[1]
        ch_types_ordered = [ordered_types[idx] for idx in sorted(unique_type_idxs)]
        for ch_type in [
            ct
            for ct in ch_types_ordered
            if ct != "stim"
            and ct in self.mne.scalings
            and ct in getattr(self.mne, "units", {})
            and ct in getattr(self.mne, "unit_scalings", {})
        ]:
            scale_bar = ScaleBar(self.mne, ch_type)
            self.mne.scalebars[ch_type] = scale_bar
            self.mne.plt.addItem(scale_bar)

            scale_bar_text = ScaleBarText(self.mne, ch_type)
            self.mne.scalebar_texts[ch_type] = scale_bar_text
            self.mne.plt.addItem(scale_bar_text)

        self._set_scalebars_visible(self.mne.scalebars_visible)

    def _update_scalebar_x_positions(self):
        if self.mne.scalebars_visible:
            for scalebar in self.mne.scalebars.values():
                scalebar.update_x_position()

            for scalebar_text in self.mne.scalebar_texts.values():
                scalebar_text.update_x_position()

    def _update_scalebar_y_positions(self):
        if self.mne.scalebars_visible:
            for scalebar in self.mne.scalebars.values():
                scalebar.update_y_position()

            for scalebar_text in self.mne.scalebar_texts.values():
                scalebar_text.update_y_position()

    def _update_scalebar_values(self):
        for scalebar_text in self.mne.scalebar_texts.values():
            scalebar_text.update_value()

    def _update_ch_spinbox_values(self):
        if self.mne.fig_settings is not None:
            self.mne.fig_settings._update_spinbox_values(ch_type="all", source="all")

    def _set_scalebars_visible(self, visible):
        for scalebar in self.mne.scalebars.values():
            scalebar.setVisible(visible)

        for scalebar_text in self.mne.scalebar_texts.values():
            scalebar_text.setVisible(visible)

        self._update_scalebar_y_positions()

    def _toggle_scalebars(self):
        self.mne.scalebars_visible = not self.mne.scalebars_visible
        self._set_scalebars_visible(self.mne.scalebars_visible)

    def _overview_mode_changed(self, new_mode):
        self.mne.overview_mode = new_mode
        self._save_setting("overview_mode", new_mode)
        self.mne.overview_bar.set_background()
        if not self.mne.overview_bar.isVisible():
            self._toggle_overview_bar()

    def _overview_radio_clicked(self, checked=False, *, menu, new_mode):
        menu.close()
        self._overview_mode_changed(new_mode=new_mode)

    def scale_all(self, checked=False, *, step, update_spinboxes=True):
        """Scale all traces by multiplying with step."""
        self.mne.scale_factor *= step

        # Reapply clipping if necessary
        if self.mne.clipping is not None:
            self._update_data()

        # Scale Traces (by scaling the Item, not the data)
        for line in self.mne.traces:
            line.update_scale()

        # Update Scalebars
        self._update_scalebar_values()

        # Update spinboxes in settings dialog
        if update_spinboxes:
            self._update_ch_spinbox_values()

    def hscroll(self, step):
        """Scroll horizontally by step."""
        if isinstance(step, str):
            if step in ("-full", "+full"):
                rel_step = self.mne.duration
                if step == "-full":
                    rel_step = rel_step * -1
            else:
                assert step in ("left", "right")
                if self.mne.is_epochs:
                    rel_step = self.mne.duration / self.mne.n_epochs
                else:
                    rel_step = 0.25 * self.mne.duration
                if step == "left":
                    rel_step = rel_step * -1
        else:
            if self.mne.is_epochs:
                rel_step = np.sign(step) * self.mne.duration / self.mne.n_epochs
            else:
                rel_step = step * self.mne.duration / self.mne.scroll_sensitivity
        del step

        # Get current range and add step to it
        xmin, xmax = (i + rel_step for i in self.mne.viewbox.viewRange()[0])

        if xmin < 0:
            xmin = 0
            xmax = xmin + self.mne.duration
        elif xmax > self.mne.xmax:
            xmax = self.mne.xmax
            xmin = xmax - self.mne.duration

        self.mne.plt.setXRange(xmin, xmax, padding=0)

    def vscroll(self, step):
        """Scroll vertically by step."""
        if self.mne.fig_selection is not None:
            if step == "+full":
                step = 1
            elif step == "-full":
                step = -1
            else:
                step = int(step)
            self.mne.fig_selection._scroll_selection(step)
        elif self.mne.butterfly:
            return
        else:
            # Get current range and add step to it
            if step == "+full":
                step = self.mne.n_channels
            elif step == "-full":
                step = -self.mne.n_channels
            ymin, ymax = (i + step for i in self.mne.viewbox.viewRange()[1])

            if ymin < 0:
                ymin = 0
                ymax = self.mne.n_channels + 1
            elif ymax > self.mne.ymax:
                ymax = self.mne.ymax
                ymin = ymax - self.mne.n_channels - 1

            self.mne.plt.setYRange(ymin, ymax, padding=0)

    def change_duration(self, checked=False, *, step):
        """Change duration by step."""
        xmin, xmax = self.mne.viewbox.viewRange()[0]

        if self.mne.is_epochs:
            # use the length of one epoch as duration change
            min_dur = len(self.mne.inst.times) / self.mne.info["sfreq"]
            step_dir = 1 if step > 0 else -1
            rel_step = min_dur * step_dir
            self.mne.n_epochs = np.clip(
                self.mne.n_epochs + step_dir, 1, len(self.mne.inst)
            )
        else:
            # never show fewer than 3 samples
            min_dur = 3 * np.diff(self.mne.inst.times[:2])[0]
            rel_step = self.mne.duration * step

        xmax += rel_step

        if xmax - xmin < min_dur:
            xmax = xmin + min_dur

        if xmax > self.mne.xmax:
            diff = xmax - self.mne.xmax
            xmax = self.mne.xmax
            xmin -= diff

        if xmin < 0:
            xmin = 0

        self.mne.ax_hscroll.update_duration()
        self.mne.plt.setXRange(xmin, xmax, padding=0)

    def change_nchan(self, checked=False, *, step):
        """Change number of channels by step."""
        if not self.mne.butterfly:
            if step == "+full":
                step = self.mne.n_channels
            elif step == "-full":
                step = -self.mne.n_channels
            ymin, ymax = self.mne.viewbox.viewRange()[1]
            ymax += step
            if ymax > self.mne.ymax:
                ymax = self.mne.ymax
                ymin -= step

            if ymin < 0:
                ymin = 0

            if ymax - ymin <= 2:
                ymax = ymin + 2

            self.mne.ax_vscroll.update_nchan()
            self.mne.plt.setYRange(ymin, ymax, padding=0)

        if self.mne.fig_settings is not None:
            self.mne.fig_settings._update_spinbox_values(ch_type="all", source="chans")

    def _remove_vline(self):
        if self.mne.vline is not None:
            if self.mne.is_epochs:
                for vline in self.mne.vline:
                    self.mne.plt.removeItem(vline)
            else:
                self.mne.plt.removeItem(self.mne.vline)

        self.mne.vline = None
        self.mne.vline_visible = False
        self.mne.overview_bar.update_vline()

    def _get_vline_times(self, t):
        rel_time = t % self.mne.epoch_dur
        abs_time = self.mne.times[0]
        ts = np.arange(self.mne.n_epochs) * self.mne.epoch_dur + abs_time + rel_time

        return ts

    def _vline_slot(self, orig_vline):
        if self.mne.is_epochs:
            ts = self._get_vline_times(orig_vline.value())
            for vl, xt in zip(self.mne.vline, ts):
                if vl != orig_vline:
                    vl.setPos(xt)
        self.mne.overview_bar.update_vline()

    def _add_vline(self, t):
        if self.mne.is_epochs:
            ts = self._get_vline_times(t)

            # Add vline if None
            if self.mne.vline is None:
                self.mne.vline = list()
                for xt in ts:
                    epo_idx = np.clip(
                        np.searchsorted(self.mne.boundary_times, xt) - 1,
                        0,
                        len(self.mne.inst),
                    )
                    bmin, bmax = self.mne.boundary_times[epo_idx : epo_idx + 2]
                    # Avoid off-by-one-error at bmax for VlineLabel
                    bmax -= 1 / self.mne.info["sfreq"]
                    vl = VLine(self.mne, xt, bounds=(bmin, bmax))
                    # Should only be emitted when dragged
                    vl.sigPositionChangeFinished.connect(self._vline_slot)
                    self.mne.vline.append(vl)
                    self.mne.plt.addItem(vl)
            else:
                for vl, xt in zip(self.mne.vline, ts):
                    vl.setPos(xt)
        else:
            if self.mne.vline is None:
                self.mne.vline = VLine(self.mne, t, bounds=(0, self.mne.xmax))
                self.mne.vline.sigPositionChangeFinished.connect(self._vline_slot)
                self.mne.plt.addItem(self.mne.vline)
            else:
                self.mne.vline.setPos(t)

        self.mne.vline_visible = True
        self.mne.overview_bar.update_vline()

    def _mouse_moved(self, pos):
        """Show Crosshair if enabled at mouse move."""
        if self.mne.crosshair_enabled:
            if self.mne.plt.sceneBoundingRect().contains(pos):
                mousePoint = self.mne.viewbox.mapSceneToView(pos)
                x, y = mousePoint.x(), mousePoint.y()
                if 0 <= x <= self.mne.xmax and 0 <= y <= self.mne.ymax:
                    if not self.mne.crosshair:
                        self.mne.crosshair = Crosshair(self.mne)
                        self.mne.plt.addItem(self.mne.crosshair, ignoreBounds=True)

                    # Get ypos from trace
                    trace = [
                        tr
                        for tr in self.mne.traces
                        if tr.ypos - 0.5 < y < tr.ypos + 0.5
                    ]
                    if len(trace) == 1:
                        trace = trace[0]
                        idx = np.searchsorted(self.mne.times, x)
                        if self.mne.data_precomputed:
                            data = self.mne.data[trace.order_idx]
                        else:
                            data = self.mne.data[trace.range_idx]
                        yvalue = data[idx]
                        yshown = yvalue + trace.ypos
                        self.mne.crosshair.set_data(x, yshown)

                        # relative x for epochs
                        if self.mne.is_epochs:
                            rel_idx = idx % len(self.mne.inst.times)
                            x = self.mne.inst.times[rel_idx]

                        # negative because plot is inverted for Y
                        inv_norm = _get_channel_scaling(self, trace.ch_type) * -1
                        label = (
                            f"{_simplify_float(yvalue * inv_norm)} "
                            f"{self.mne.units[trace.ch_type]}"
                        )
                        self.statusBar().showMessage(f"x={x:.3f} s, y={label}")

    def _toggle_crosshair(self, checked=None):
        if checked is None:
            checked = not self.mne.crosshair_enabled
        self.mne.crosshair_enabled = checked
        self.mne.crosshair_action.setChecked(checked)
        if self.mne.crosshair:
            self.mne.plt.removeItem(self.mne.crosshair)
            self.mne.crosshair = None
        if not checked:
            self.statusBar().clearMessage()

    def _xrange_changed(self, _, xrange):
        # Update data
        if self.mne.is_epochs:
            if self.mne.vline is not None:
                rel_vl_t = (
                    self.mne.vline[0].value()
                    - self.mne.boundary_times[self.mne.epoch_idx][0]
                )

            # Depends on only allowing xrange showing full epochs
            boundary_idxs = np.searchsorted(self.mne.midpoints, xrange)
            self.mne.epoch_idx = np.arange(*boundary_idxs)

            # Update colors
            for trace in self.mne.traces:
                trace.update_color()

            # Update vlines
            if self.mne.vline is not None:
                for bmin, bmax, vl in zip(
                    self.mne.boundary_times[self.mne.epoch_idx],
                    self.mne.boundary_times[self.mne.epoch_idx + 1],
                    self.mne.vline,
                ):
                    # Avoid off-by-one-error at bmax for VlineLabel
                    bmax -= 1 / self.mne.info["sfreq"]
                    vl.setBounds((bmin, bmax))
                    vl.setValue(bmin + rel_vl_t)

        self.mne.t_start = xrange[0]
        self.mne.duration = xrange[1] - xrange[0]

        self._redraw(update_data=True)

        # Update Time-Bar
        self.mne.ax_hscroll.update_value(xrange[0])

        # Update Overview-Bar
        self.mne.overview_bar.update_viewrange()

        # Update Scalebars
        self._update_scalebar_x_positions()

        # Update annotations
        self._update_regions_visible()

    def _yrange_changed(self, _, yrange):
        if not self.mne.butterfly:
            if not self.mne.fig_selection:
                # Update picks and data
                self.mne.ch_start = np.clip(
                    round(yrange[0]), 0, len(self.mne.ch_order) - self.mne.n_channels
                )
                self.mne.n_channels = round(yrange[1] - yrange[0] - 1)
                self._update_picks()
                # Update Channel-Bar
                self.mne.ax_vscroll.update_value(self.mne.ch_start)
            self._update_data()

        # Update Overview-Bar
        self.mne.overview_bar.update_viewrange()

        # Update Scalebars
        self._update_scalebar_y_positions()

        off_traces = [tr for tr in self.mne.traces if tr.ch_idx not in self.mne.picks]
        add_idxs = [
            p for p in self.mne.picks if p not in [tr.ch_idx for tr in self.mne.traces]
        ]

        # Update range_idx for traces which just shifted in y-position
        for trace in [tr for tr in self.mne.traces if tr not in off_traces]:
            trace.update_range_idx()

        # Update number of traces.
        trace_diff = len(self.mne.picks) - len(self.mne.traces)

        # Remove unnecessary traces.
        if trace_diff < 0:
            # Only remove from traces not in picks.
            remove_traces = off_traces[: abs(trace_diff)]
            for trace in remove_traces:
                trace.remove()
                off_traces.remove(trace)

        # Add new traces if necessary.
        if trace_diff > 0:
            # Make copy to avoid skipping iteration.
            idxs_copy = add_idxs.copy()
            for aidx in idxs_copy[:trace_diff]:
                DataTrace(self, aidx)
                add_idxs.remove(aidx)

        # Update data of traces outside of yrange (reuse remaining trace-items)
        for trace, ch_idx in zip(off_traces, add_idxs):
            trace.set_ch_idx(ch_idx)
            trace.update_color()
            trace.update_data()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # DATA HANDLING
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def _apply_downsampling(self):
        """
        Get ds-factor and apply ds with one of multiple methods.

        The methods are taken from PlotDataItem in pyqtgraph
        and adjusted to multi-channel data.
        """
        # Get Downsampling-Factor
        # Auto-Downsampling from pyqtgraph
        if self.mne.downsampling == "auto":
            ds = 1
            if all([hasattr(self.mne, a) for a in ["viewbox", "times"]]):
                vb = self.mne.viewbox
                if vb is not None:
                    view_range = vb.viewRect()
                else:
                    view_range = None
                if view_range is not None and len(self.mne.times) > 1:
                    dx = float(self.mne.times[-1] - self.mne.times[0]) / (
                        len(self.mne.times) - 1
                    )
                    if dx != 0.0:
                        x0 = view_range.left() / dx
                        x1 = view_range.right() / dx
                        width = vb.width()
                        if width != 0.0:
                            # Auto-Downsampling with 5 samples per pixel
                            ds = int(max(1, (x1 - x0) / (width * 5)))
        else:
            ds = self.mne.downsampling

        # Apply Downsampling
        if ds not in [None, 1]:
            times = self.mne.times
            data = self.mne.data
            n_ch = data.shape[0]

            if self.mne.ds_method == "subsample":
                times = times[::ds]
                data = data[:, ::ds]

            elif self.mne.ds_method == "mean":
                n = len(times) // ds
                # start of x-values
                # try to select a somewhat centered point
                stx = ds // 2
                times = times[stx : stx + n * ds : ds]
                rs_data = data[:, : n * ds].reshape(n_ch, n, ds)
                data = rs_data.mean(axis=2)

            elif self.mne.ds_method == "peak":
                n = len(times) // ds
                # start of x-values
                # try to select a somewhat centered point
                stx = ds // 2

                x1 = np.empty((n, 2))
                x1[:] = times[stx : stx + n * ds : ds, np.newaxis]
                times = x1.reshape(n * 2)

                y1 = np.empty((n_ch, n, 2))
                y2 = data[:, : n * ds].reshape((n_ch, n, ds))
                y1[:, :, 0] = y2.max(axis=2)
                y1[:, :, 1] = y2.min(axis=2)
                data = y1.reshape((n_ch, n * 2))

            self.mne.times, self.mne.data = times, data

    def _show_process(self, message):
        if self.mne.load_progressbar.isVisible():
            self.mne.load_progressbar.hide()
            self.mne.load_prog_label.hide()
        self.statusBar().showMessage(message)

    def _precompute_finished(self):
        self.statusBar().showMessage("Loading Finished", 5)
        self.mne.data_precomputed = True

        if self.mne.overview_mode == "zscore":
            # Show loaded overview image
            self.mne.overview_bar.set_background()

        if self._rerun_load_thread:
            self._rerun_load_thread = False
            self._init_precompute()

    def _init_precompute(self):
        # Remove previously loaded data
        self.mne.data_precomputed = False
        if all([hasattr(self.mne, st) for st in ["global_data", "global_times"]]):
            del self.mne.global_data, self.mne.global_times
        gc.collect()

        if self.mne.precompute == "auto":
            self.mne.enable_precompute = self._check_space_for_precompute()
        elif isinstance(self.mne.precompute, bool):
            self.mne.enable_precompute = self.mne.precompute

        if self.mne.enable_precompute:
            # Start precompute thread
            self.mne.load_progressbar.show()
            self.mne.load_prog_label.show()
            self.load_thread.start()

    def _rerun_precompute(self):
        if self.load_thread.isRunning():
            self._rerun_load_thread = True
        else:
            self._init_precompute()

    def _check_space_for_precompute(self):
        try:
            import psutil
        except ImportError:
            logger.info(
                "Free RAM space could not be determined because"
                '"psutil" is not installed. '
                "Setting precompute to False."
            )
            return False
        else:
            if self.mne.is_epochs:
                files = [self.mne.inst.filename]
            else:
                files = self.mne.inst.filenames
            if files[0] is not None:
                # Get disk-space of raw-file(s)
                disk_space = 0
                for fn in files:
                    disk_space += getsize(fn)

                # Determine expected RAM space based on orig_format
                fmt_multipliers = {"double": 1, "single": 2, "int": 2, "short": 4}

                # Epochs and ICA don't have this attribute, assume single
                # on disk
                fmt = getattr(self.mne.inst, "orig_format", "single")
                # Apply size change to 64-bit float in memory
                # (* 2 because when loading data will be loaded into a copy
                # of self.mne.inst._data to apply processing.
                expected_ram = disk_space * fmt_multipliers[fmt] * 2
            else:
                expected_ram = sys.getsizeof(self.mne.inst._data)

            # Get available RAM
            free_ram = psutil.virtual_memory().free

            expected_ram_str = sizeof_fmt(expected_ram)
            free_ram_str = sizeof_fmt(free_ram)
            left_ram_str = sizeof_fmt(free_ram - expected_ram)

            if expected_ram < free_ram:
                logger.debug(
                    "The data precomputed for visualization takes "
                    f"{expected_ram_str} with {left_ram_str} of "
                    f"RAM left."
                )
                return True
            else:
                logger.debug(
                    f"The precomputed data with {expected_ram_str} "
                    f"will surpass your current {free_ram_str} "
                    f"of free RAM.\n"
                    "Thus precompute will be set to False.\n"
                    "(If you want to precompute nevertheless, "
                    'then set precompute to True instead of "auto")'
                )
                return False

    def _process_data(self, *args, **kwargs):
        data = super()._process_data(*args, **kwargs)

        # Invert Data to be displayed from top on inverted Y-Axis
        data *= -1

        return data

    def _update_data(self):
        if self.mne.data_precomputed:
            # get start/stop-samples
            start, stop = self._get_start_stop()
            self.mne.times = self.mne.global_times[start:stop]
            self.mne.data = self.mne.global_data[:, start:stop]

            # remove DC locally
            if self.mne.remove_dc:
                self.mne.data = self.mne.data - np.nanmean(
                    self.mne.data, axis=1, keepdims=True
                )
        else:
            # While data is not precomputed get data only from shown range and
            # process only those.
            super()._update_data()

        # Initialize decim
        self.mne.decim_data = np.ones_like(self.mne.picks)
        data_picks_mask = np.isin(self.mne.picks, self.mne.picks_data)
        self.mne.decim_data[data_picks_mask] = self.mne.decim

        # Apply clipping
        if self.mne.clipping == "clamp":
            self.mne.data = np.clip(self.mne.data, -0.5, 0.5)
        elif self.mne.clipping is not None:
            self.mne.data = self.mne.data.copy()
            self.mne.data[
                abs(self.mne.data * self.mne.scale_factor) > self.mne.clipping
            ] = np.nan

        # Apply Downsampling (if enabled)
        self._apply_downsampling()

    def _get_zscore(self, data):
        # Reshape data to reasonable size for display
        screen_geometry = _screen_geometry(self)
        if screen_geometry is None:
            max_pixel_width = 3840  # default=UHD
        else:
            max_pixel_width = screen_geometry.width()
        collapse_by = data.shape[1] // max_pixel_width
        data = data[:, : max_pixel_width * collapse_by]
        if collapse_by > 0:
            data = data.reshape(data.shape[0], max_pixel_width, collapse_by)
            data = data.mean(axis=2)
        z = zscore(data, axis=1)
        if z.size > 0:
            zmin = np.min(z, axis=1)
            zmax = np.max(z, axis=1)

            # Convert into RGBA
            zrgba = np.empty((*z.shape, 4))
            for row_idx, row in enumerate(z):
                for col_idx, value in enumerate(row):
                    if math.isnan(value):
                        value = 0
                    if value == 0:
                        rgba = [0, 0, 0, 0]
                    elif value < 0:
                        alpha = int(255 * value / abs(zmin[row_idx]))
                        rgba = [0, 0, 255, alpha]
                    else:
                        alpha = int(255 * value / zmax[row_idx])
                        rgba = [255, 0, 0, alpha]

                    zrgba[row_idx, col_idx] = rgba

            zrgba = np.require(zrgba, np.uint8, "C")

            self.mne.zscore_rgba = zrgba

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # ANNOTATIONS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def _add_region(
        self, plot_onset, duration, description, *, ch_names=None, region=None
    ):
        if not region:
            region = AnnotRegion(
                self.mne,
                description=description,
                ch_names=ch_names,
                values=(plot_onset, plot_onset + duration),
                weakmain=weakref.ref(self),
            )
        # Add region to list and plot
        self.mne.regions.append(region)

        # Connect signals of region
        region.regionChangeFinished.connect(self._region_changed)
        region.gotSelected.connect(self._region_selected)
        region.removeRequested.connect(self._remove_region)
        self.mne.viewbox.sigYRangeChanged.connect(region.update_label_pos)
        region.update_label_pos()
        return region

    def _remove_region(self, region, from_annot=True):
        # Remove from shown regions
        if region.label_item in self.mne.plt.items:
            self.mne.plt.removeItem(region.label_item)
        if region in self.mne.plt.items:
            self.mne.plt.removeItem(region)

        # Remove from all regions
        if region in self.mne.regions:
            self.mne.regions.remove(region)

        # Reset selected region
        if region == self.mne.selected_region:
            self.mne.selected_region = None
            # disable, reset start/stop doubleSpinBox until another region is selected
            self.mne.fig_annotation.start_bx.setEnabled(False)
            self.mne.fig_annotation.stop_bx.setEnabled(False)
            with QSignalBlocker(self.mne.fig_annotation.start_bx):
                self.mne.fig_annotation.start_bx.setValue(0)
            with QSignalBlocker(self.mne.fig_annotation.stop_bx):
                self.mne.fig_annotation.stop_bx.setValue(1 / self.mne.info["sfreq"])

        # Remove from annotations
        if from_annot:
            idx = self._get_onset_idx(region.getRegion()[0])
            self.mne.inst.annotations.delete(idx)

        # Update Overview-Bar
        self.mne.overview_bar.update_annotations()

    def _region_selected(self, region):
        old_region = self.mne.selected_region
        # Remove selected-status from old region
        if old_region and old_region != region:
            old_region.select(False)
        self.mne.selected_region = region
        self.mne.fig_annotation.update_values(region)

    def _get_onset_idx(self, plot_onset):
        onset = _sync_onset(self.mne.inst, plot_onset, inverse=True)
        idx = np.where(self.mne.inst.annotations.onset == onset)[0][0]
        return idx

    def _region_changed(self, region):
        rgn = region.getRegion()
        region.select(True)
        idx = self._get_onset_idx(region.old_onset)
        # update Spinboxes of Annot-Dock
        self.mne.fig_annotation.update_values(region)
        # edit inst.annotations
        onset = _sync_onset(self.mne.inst, rgn[0], inverse=True)
        self.mne.inst.annotations.onset[idx] = onset
        self.mne.inst.annotations.duration[idx] = rgn[1] - rgn[0]
        _merge_annotations(
            onset,
            onset + rgn[1] - rgn[0],
            region.description,
            self.mne.inst.annotations,
        )
        # update overview-bar
        self.mne.overview_bar.update_annotations()

    def _draw_annotations(self):
        # All regions are constantly added to the Scene and handled by Qt
        # which is faster than handling adding/removing in Python.
        pass

    def _init_annot_mode(self):
        self.mne.annotations_visible = True
        self.mne.new_annotation_labels = self._get_annotation_labels()
        if len(self.mne.new_annotation_labels) > 0:
            self.mne.current_description = self.mne.new_annotation_labels[0]
        else:
            self.mne.current_description = None
        self._setup_annotation_colors()
        self.mne.regions = list()
        self.mne.selected_region = None

        # Initialize Annotation-Dock
        existing_dock = getattr(self.mne, "fig_annotation", None)
        if existing_dock is None:
            self.mne.fig_annotation = AnnotationDock(self)
            self.addDockWidget(Qt.TopDockWidgetArea, self.mne.fig_annotation)
            self.mne.fig_annotation.setVisible(False)
            self.mne.fig_annotation.start_bx.setEnabled(False)
            self.mne.fig_annotation.stop_bx.setEnabled(False)

        # Add annotations as regions
        for annot in self.mne.inst.annotations:
            plot_onset = _sync_onset(self.mne.inst, annot["onset"])
            duration = annot["duration"]
            description = annot["description"]
            ch_names = annot["ch_names"] if "ch_names" in annot else None
            region = self._add_region(
                plot_onset, duration, description, ch_names=ch_names
            )
            region.update_visible(False)

        # Initialize showing annotation widgets
        self._change_annot_mode()

    def _change_annot_mode(self):
        if not self.mne.annotation_mode:
            # Reset Widgets in Annotation-Figure
            self.mne.fig_annotation.reset()

        # Show Annotation-Dock if activated.
        self.mne.fig_annotation.setVisible(self.mne.annotation_mode)

        # Make Regions movable if activated and move into foreground
        for region in self.mne.regions:
            region.setMovable(self.mne.annotation_mode)
            if self.mne.annotation_mode:
                region.setZValue(
                    2 if region.description == self.mne.current_description else 1
                )
            else:
                region.setZValue(0)

        # Add/Remove selection-rectangle.
        if self.mne.selected_region:
            self.mne.selected_region.select(self.mne.annotation_mode)

    def _toggle_annotation_fig(self):
        if not self.mne.is_epochs:
            self.mne.annotation_mode = not self.mne.annotation_mode
            self._change_annot_mode()

    def _update_regions_visible(self):
        if self.mne.is_epochs:
            return
        start = self.mne.t_start
        stop = start + self.mne.duration
        for region in self.mne.regions:
            if self.mne.visible_annotations[region.description]:
                rgn = region.getRegion()
                # Avoid NumPy bool here
                visible = bool(rgn[0] <= stop and rgn[1] >= start)
            else:
                visible = False
            region.update_visible(visible)
        self.mne.overview_bar.update_annotations()

    def _set_annotations_visible(self, visible):
        for descr in self.mne.visible_annotations:
            self.mne.visible_annotations[descr] = visible
        self._update_regions_visible()

    def _toggle_annotations(self):
        self.mne.annotations_visible = not self.mne.annotations_visible
        self._set_annotations_visible(self.mne.annotations_visible)

    def _apply_update_projectors(self, toggle_all=False):
        if toggle_all:
            on = self.mne.projs_on
            applied = self.mne.projs_active
            value = False if all(on) else True
            new_state = np.full_like(on, value)
            # Always activate applied projections
            new_state[applied] = True
            self.mne.projs_on = new_state
        self._update_projector()
        # If data was precomputed it needs to be precomputed again.
        self._rerun_precompute()
        self._redraw()

    def _toggle_proj_fig(self):
        if self.mne.fig_proj is None:
            ProjDialog(self, name="fig_proj")
        else:
            self.mne.fig_proj.close()
            self.mne.fig_proj = None

    def _toggle_all_projs(self):
        if self.mne.fig_proj is None:
            self._apply_update_projectors(toggle_all=True)
        else:
            self.mne.fig_proj.toggle_all()

    def _toggle_whitening(self):
        if self.mne.noise_cov is not None:
            super()._toggle_whitening()
            # If data was precomputed it needs to be precomputed again.
            self._rerun_precompute()
            self._redraw()

    def _toggle_settings_fig(self):
        if self.mne.fig_settings is None:
            SettingsDialog(self, name="fig_settings")
        else:
            self.mne.fig_settings.close()
            self.mne.fig_settings = None

    def _toggle_help_fig(self):
        if self.mne.fig_help is None:
            HelpDialog(self, name="fig_help")
        else:
            self.mne.fig_help.close()
            self.mne.fig_help = None

    def _set_butterfly(self, butterfly):
        self.mne.butterfly = butterfly
        self._update_picks()
        self._update_data()

        if butterfly and self.mne.fig_selection is not None:
            self.mne.selection_ypos_dict.clear()
            selections_dict = self._make_butterfly_selections_dict()
            for idx, picks in enumerate(selections_dict.values()):
                for pick in picks:
                    self.mne.selection_ypos_dict[pick] = idx + 1
            ymax = len(selections_dict) + 1
            self.mne.ymax = ymax
            self.mne.plt.setLimits(yMax=ymax)
            self.mne.plt.setYRange(0, ymax, padding=0)
        elif butterfly:
            ymax = len(self.mne.butterfly_type_order) + 1
            self.mne.ymax = ymax
            self.mne.plt.setLimits(yMax=ymax)
            self.mne.plt.setYRange(0, ymax, padding=0)
        else:
            self.mne.ymax = len(self.mne.ch_order) + 1
            self.mne.plt.setLimits(yMax=self.mne.ymax)
            self.mne.plt.setYRange(
                self.mne.ch_start,
                self.mne.ch_start + self.mne.n_channels + 1,
                padding=0,
            )

        if self.mne.fig_selection is not None:
            # Update Selection-Dialog
            self.mne.fig_selection._style_butterfly()

        # Set vertical scrollbar visible
        self.mne.ax_vscroll.setVisible(
            not butterfly or self.mne.fig_selection is not None
        )

        # update overview-bar
        self.mne.overview_bar.update_viewrange()

        # update ypos and color for butterfly-mode
        for trace in self.mne.traces:
            trace.update_color()
            trace.update_ypos()

        self._draw_traces()

        self._update_ch_spinbox_values()

    def _toggle_butterfly(self):
        if self.mne.instance_type != "ica":
            self._set_butterfly(not self.mne.butterfly)

    def _toggle_dc(self):
        self.mne.remove_dc = not self.mne.remove_dc
        self._redraw()

    def _set_events_visible(self, visible):
        for event_line in self.mne.event_lines:
            event_line.setVisible(visible)

        self.mne.overview_bar.update_events()

    def _toggle_events(self):
        if self.mne.event_nums is not None:
            self.mne.events_visible = not self.mne.events_visible
            self._set_events_visible(self.mne.events_visible)

    def _toggle_time_format(self):
        if self.mne.info["meas_date"] is not None:
            if self.mne.time_format == "float":
                self.mne.time_format = "clock"
                self.mne.time_axis.setLabel(text="Time of day")
            else:
                self.mne.time_format = "float"
                self.mne.time_axis.setLabel(text="Time", units="s")
            self._update_yaxis_labels()

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _toggle_antialiasing(self):
        self.mne.antialiasing = not self.mne.antialiasing
        self._redraw()

    def _toggle_overview_bar(self):
        visible = not self.mne.overview_bar.isVisible()
        for item in self.mne.overview_menu.actions():
            if item.text() == "Visible":
                item.setChecked(visible)
                break
        self.mne.overview_bar.setVisible(visible)
        self.mne.overview_visible = visible
        self._save_setting("overview_visible", visible)

    def _toggle_zenmode(self):
        self.mne.scrollbars_visible = not self.mne.scrollbars_visible
        for bar in [self.mne.ax_hscroll, self.mne.ax_vscroll]:
            bar.setVisible(self.mne.scrollbars_visible)
        self.statusBar().setVisible(self.mne.scrollbars_visible)
        self.mne.toolbar.setVisible(self.mne.scrollbars_visible)

    def _new_child_figure(self, fig_name, window_title, **kwargs):
        from matplotlib.figure import Figure

        fig = Figure(**kwargs)
        # Pass window title and fig_name on
        if fig_name is not None:
            fig.fig_name = fig_name
        if window_title is not None:
            fig.title = window_title
        return fig

    def _get_widget_from_mpl(self, fig):
        canvas = FigureCanvasQTAgg(fig)
        canvas.setFocusPolicy(Qt.FocusPolicy(Qt.StrongFocus | Qt.WheelFocus))
        canvas.setFocus()
        _set_window_flags(canvas)
        # Pass window title and fig_name on
        if hasattr(fig, "fig_name"):
            canvas.fig_name = fig.fig_name
        if hasattr(fig, "title"):
            canvas.title = fig.title

        return canvas

    def _get_dlg_from_mpl(self, fig):
        canvas = self._get_widget_from_mpl(fig)
        # Pass window title and fig_name on
        if hasattr(canvas, "fig_name"):
            name = canvas.fig_name
        else:
            name = None
        if hasattr(canvas, "title"):
            title = canvas.title
        else:
            title = None
        dlg = _BaseDialog(self, widget=canvas, title=title, name=name)
        dlg.show()

    def _create_ch_context_fig(self, idx):
        fig = super()._create_ch_context_fig(idx)
        if fig is not None:
            self._get_dlg_from_mpl(fig)

    def _toggle_epoch_histogram(self):
        if self.mne.is_epochs:
            fig = self._create_epoch_histogram()
            if fig is not None:
                self._get_dlg_from_mpl(fig)

    def _create_selection_fig(self):
        if not any([isinstance(fig, SelectionDialog) for fig in self.mne.child_figs]):
            SelectionDialog(self)

    def message_box(
        self,
        text,
        info_text=None,
        buttons=None,
        default_button=None,
        icon=None,
        modal=True,
    ):  # noqa: D102
        self.msg_box.setText(f'<font size="+2"><b>{text}</b></font>')
        if info_text is not None:
            self.msg_box.setInformativeText(info_text)
        if buttons is not None:
            self.msg_box.setStandardButtons(buttons)
        if default_button is not None:
            self.msg_box.setDefaultButton(default_button)
        if icon is not None:
            self.msg_box.setIcon(icon)

        # Allow interacting with message_box in test-mode.
        # Set modal=False only if no return value is expected.
        self.msg_box.setModal(False if self.test_mode else modal)
        if self.test_mode or not modal:
            self.msg_box.show()
        else:
            return self.msg_box.exec()

    def keyPressEvent(self, event):
        """Customize key press events."""
        # On MacOs additionally KeypadModifier is set when arrow-keys
        # are pressed.
        mods = event.modifiers()
        try:
            mods = int(mods)  # PyQt < 5.13
        except Exception:
            pass
        modifiers = {
            "Shift": bool(Qt.ShiftModifier & mods),
            "Ctrl": bool(Qt.ControlModifier & mods),
        }
        for key_name in self.mne.keyboard_shortcuts:
            key_dict = self.mne.keyboard_shortcuts[key_name]
            if key_dict["qt_key"] == event.key() and "slot" in key_dict:
                mod_idx = 0
                # Get modifier
                if "modifier" in key_dict:
                    mods = [modifiers[mod] for mod in modifiers]
                    if any(mods):
                        # No multiple modifiers supported yet
                        mod = [mod for mod in modifiers if modifiers[mod]][0]
                        if mod in key_dict["modifier"]:
                            mod_idx = key_dict["modifier"].index(mod)

                slot_idx = mod_idx if mod_idx < len(key_dict["slot"]) else 0
                slot = key_dict["slot"][slot_idx]

                if "parameter" in key_dict:
                    param_idx = mod_idx if mod_idx < len(key_dict["parameter"]) else 0
                    val = key_dict["parameter"][param_idx]
                    if "kw" in key_dict:
                        slot(**{key_dict["kw"]: val})
                    else:
                        slot(val)
                else:
                    slot()

                break

    def _draw_traces(self):
        # Update data in traces (=drawing traces)
        for trace in self.mne.traces:
            # Update data
            trace.update_data()

    def _get_size(self):
        inch_width = self.width() / self.logicalDpiX()
        inch_height = self.height() / self.logicalDpiY()
        logger.debug(f"Window size: {inch_width:0.1f} x {inch_height:0.1f} inches")
        return inch_width, inch_height

    def _fake_keypress(self, key, fig=None):
        fig = fig or self

        if key.isupper():
            key = key.lower()
            modifier = Qt.ShiftModifier
        elif key.startswith("shift+"):
            key = key[6:]
            modifier = Qt.ShiftModifier
        else:
            modifier = Qt.NoModifier

        # Use pytest-qt's exception-hook
        with capture_exceptions() as exceptions:
            QTest.keyPress(fig, self.mne.keyboard_shortcuts[key]["qt_key"], modifier)

        for exc in exceptions:
            raise RuntimeError(
                f"There as been an {exc[0]} inside the Qt "
                f"event loop (look above for traceback)."
            )

    def _fake_click(
        self,
        point,
        add_points=None,
        fig=None,
        ax=None,
        xform="ax",
        button=1,
        kind="press",
        modifier=None,
    ):
        add_points = add_points or list()
        # Wait until Window is fully shown.
        QTest.qWaitForWindowExposed(self)
        # Scene-Dimensions still seem to change to final state when waiting
        # for a short time.
        QTest.qWait(10)

        # Qt: right-button=2, matplotlib: right-button=3
        if button == 1:
            button = Qt.LeftButton
        else:
            button = Qt.RightButton

        # For Qt, fig or ax both would be the widget to test interaction on.
        # If View
        fig = ax or fig or self.mne.view

        if xform == "ax":
            # For Qt, the equivalent of matplotlibs transAxes
            # would be a transformation to View Coordinates.
            # But for the View top-left is (0, 0) and bottom-right is
            # (view-width, view-height).
            view_width = fig.width()
            view_height = fig.height()
            x = view_width * point[0]
            y = view_height * (1 - point[1])
            point = Point(x, y)
            for idx, apoint in enumerate(add_points):
                x2 = view_width * apoint[0]
                y2 = view_height * (1 - apoint[1])
                add_points[idx] = Point(x2, y2)

        elif xform == "data":
            # For Qt, the equivalent of matplotlibs transData
            # would be a transformation to
            # the coordinate system of the ViewBox.
            # This only works on the View (self.mne.view)
            fig = self.mne.view
            point = self.mne.viewbox.mapViewToScene(Point(*point))
            for idx, apoint in enumerate(add_points):
                add_points[idx] = self.mne.viewbox.mapViewToScene(Point(*apoint))

        elif xform == "none" or xform is None:
            if isinstance(point, tuple | list):
                point = Point(*point)
            else:
                point = Point(point)
            for idx, apoint in enumerate(add_points):
                if isinstance(apoint, tuple | list):
                    add_points[idx] = Point(*apoint)
                else:
                    add_points[idx] = Point(apoint)

        # Use pytest-qt's exception-hook
        with capture_exceptions() as exceptions:
            widget = fig.viewport() if isinstance(fig, QGraphicsView) else fig
            if kind == "press":
                # always click because most interactivity comes form
                # mouseClickEvent from pyqtgraph (just press doesn't suffice
                # here).
                _mouseClick(widget=widget, pos=point, button=button, modifier=modifier)
            elif kind == "release":
                _mouseRelease(
                    widget=widget, pos=point, button=button, modifier=modifier
                )
            elif kind == "motion":
                _mouseMove(widget=widget, pos=point, buttons=button, modifier=modifier)
            elif kind == "drag":
                _mouseDrag(
                    widget=widget,
                    positions=[point] + add_points,
                    button=button,
                    modifier=modifier,
                )

        for exc in exceptions:
            raise RuntimeError(
                f"There as been an {exc[0]} inside the Qt "
                f"event loop (look above for traceback)."
            )

        # Waiting some time for events to be processed.
        QTest.qWait(50)

    def _fake_scroll(self, x, y, step, fig=None):
        # QTest doesn't support simulating scrolling-wheel
        self.vscroll(step)

    def _click_ch_name(self, ch_index, button):
        self.mne.channel_axis.repaint()
        # Wait because channel-axis may need time
        # (came up with test_epochs::test_plot_epochs_clicks)
        QTest.qWait(100)
        if not self.mne.butterfly:
            ch_name = str(self.mne.ch_names[self.mne.picks[ch_index]])
            xrange, yrange = self.mne.channel_axis.ch_texts[ch_name]
            x = np.mean(xrange)
            y = np.mean(yrange)

            self._fake_click((x, y), fig=self.mne.view, button=button, xform="none")

    def _resize_by_factor(self, factor):
        pass

    def _get_ticklabels(self, orientation):
        if orientation == "x":
            ax = self.mne.time_axis
        else:
            ax = self.mne.channel_axis

        return list(ax.get_labels())

    def _get_scale_bar_texts(self):
        return tuple(t.toPlainText() for t in self.mne.scalebar_texts.values())

    def show(self):  # noqa: D102
        # Set raise_window like matplotlib if possible
        super().show()
        _qt_raise_window(self)

    def _close_event(self, fig=None):
        """Force calling of the MPL figure close event."""
        fig = fig or self
        if hasattr(fig, "canvas"):
            try:
                fig.canvas.close_event()
            except ValueError:  # old mpl with Qt
                pass  # pragma: no cover
        else:
            fig.close()

    def _check_close(self):
        """Close annotations-mode before closing the browser."""
        if self.mne.annotation_mode:
            self._toggle_annotation_fig()
        else:
            self.close()

    def closeEvent(self, event):
        """Customize close event."""
        event.accept()
        if hasattr(self, "mne"):
            # Explicit disconnects to avoid reference cycles that gc can't
            # properly resolve ()
            if hasattr(self.mne, "plt"):
                _disconnect(self.mne.plt.sigXRangeChanged)
                _disconnect(self.mne.plt.sigYRangeChanged)
            if hasattr(self.mne, "toolbar"):
                for action in self.mne.toolbar.actions():
                    allow_error = action.text() == ""
                    _disconnect(action.triggered, allow_error=allow_error)
            # Save settings going into QSettings.
            for qsetting in qsettings_params:
                value = getattr(self.mne, qsetting)
                self._save_setting(qsetting, value)
            for attr in (
                "keyboard_shortcuts",
                "traces",
                "plt",
                "toolbar",
                "fig_annotation",
            ):
                if hasattr(self.mne, attr):
                    delattr(self.mne, attr)
            if hasattr(self.mne, "child_figs"):
                for fig in self.mne.child_figs:
                    fig.close()
                self.mne.child_figs.clear()
            for attr in ("traces", "event_lines", "regions"):
                getattr(self.mne, attr, []).clear()
            if getattr(self.mne, "vline", None) is not None:
                if self.mne.is_epochs:
                    for vl in self.mne.vline:
                        _disconnect(vl.sigPositionChangeFinished, allow_error=True)
                    self.mne.vline.clear()
                else:
                    _disconnect(
                        self.mne.vline.sigPositionChangeFinished, allow_error=True
                    )
        if getattr(self, "load_thread", None) is not None:
            self.load_thread.clean()
            self.load_thread = None

        # Remove self from browser_instances in globals
        if self in _browser_instances:
            _browser_instances.remove(self)
        self._close(event)
        self.gotClosed.emit()
        # Make sure it gets deleted after it was closed.
        self.deleteLater()
        self._closed = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.mne.fig_settings is not None:
            self.mne.fig_settings._update_spinbox_values(
                source="resize_event", ch_type="all"
            )

    def _fake_click_on_toolbar_action(self, action_name, wait_after=500):
        """Trigger event associated with action 'action_name' in toolbar."""
        for action in self.mne.toolbar.actions():
            if not action.isSeparator():
                if action.iconText() == action_name:
                    action.trigger()
                    break
        else:
            raise ValueError(f"action_name={repr(action_name)} not found")
        QTest.qWait(wait_after)

    def _qicon(self, name):
        # Try to pull from the theme first but fall back to the local one
        kind = "dark" if self.mne.dark else "light"
        path = Path(__file__).parent / "icons" / kind / "actions" / f"{name}.svg"
        path = path.resolve(strict=True)
        return QIcon.fromTheme(name, QIcon(str(path)))


def _get_n_figs():
    # Wait for a short time to let the Qt-loop clean up
    QTest.qWait(100)
    return len(
        [window for window in QApplication.topLevelWindows() if window.isVisible()]
    )


def _close_all():
    if len(QApplication.topLevelWindows()) > 0:
        QApplication.closeAllWindows()


# mouse testing functions adapted from pyqtgraph
# (pyqtgraph.tests.ui_testing.py)
def _mousePress(widget, pos, button, modifier=None):
    if modifier is None:
        modifier = Qt.KeyboardModifier.NoModifier
    event = QMouseEvent(
        QEvent.Type.MouseButtonPress, pos, button, Qt.MouseButton.NoButton, modifier
    )
    QApplication.sendEvent(widget, event)


def _mouseRelease(widget, pos, button, modifier=None):
    if modifier is None:
        modifier = Qt.KeyboardModifier.NoModifier
    event = QMouseEvent(
        QEvent.Type.MouseButtonRelease, pos, button, Qt.MouseButton.NoButton, modifier
    )
    QApplication.sendEvent(widget, event)


def _mouseMove(widget, pos, buttons=None, modifier=None):
    if buttons is None:
        buttons = Qt.MouseButton.NoButton
    if modifier is None:
        modifier = Qt.KeyboardModifier.NoModifier
    event = QMouseEvent(
        QEvent.Type.MouseMove, pos, Qt.MouseButton.NoButton, buttons, modifier
    )
    QApplication.sendEvent(widget, event)


def _mouseClick(widget, pos, button, modifier=None):
    _mouseMove(widget, pos)
    _mousePress(widget, pos, button, modifier)
    _mouseRelease(widget, pos, button, modifier)


def _mouseDrag(widget, positions, button, modifier=None):
    _mouseMove(widget, positions[0])
    _mousePress(widget, positions[0], button, modifier)
    # Delay for 10 ms for drag to be recognized.
    QTest.qWait(10)
    for pos in positions[1:]:
        _mouseMove(widget, pos, button, modifier)
    _mouseRelease(widget, positions[-1], button, modifier)


# modified from: https://github.com/pyvista/pyvistaqt
def _setup_ipython(ipython=None):
    # ipython magic
    if scooby.in_ipython():
        from IPython import get_ipython

        ipython = get_ipython()
        ipython.run_line_magic("gui", "qt")
        from IPython.external.qt_for_kernel import QtGui

        QtGui.QApplication.instance()
    return ipython


def _init_browser(**kwargs):
    _setup_ipython()
    setConfigOption("enableExperimental", True)
    app_kwargs = dict()
    if kwargs.get("splash", False):
        app_kwargs["splash"] = "Initializing mne-qt-browser..."
    out = _init_mne_qtapp(pg_app=True, **app_kwargs)
    if "splash" in app_kwargs:
        kwargs["splash"] = out[1]  # returned as second element
    browser = MNEQtBrowser(**kwargs)

    return browser
