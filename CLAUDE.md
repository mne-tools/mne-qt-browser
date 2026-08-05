# mne-qt-browser

PyQtGraph-based backend for MNE-Python's raw/epochs/ICA browser (`mne.io.Raw.plot()` and
friends once `mne.viz.set_browser_backend("qt")` is active). MNE-Python owns the data and
the browser API; this package owns everything you see.

## Conventions

This package is a subsidiary of MNE-Python and generally follows its conventions —
naming, numpydoc style with its local deviations, absolute and lazily-nested imports,
deprecation policy, compact tests, license rules for adapted code. See
[MNE-Python's AGENTS.md](https://github.com/mne-tools/mne-python/blob/main/AGENTS.md),
which its [CLAUDE.md](https://github.com/mne-tools/mne-python/blob/main/CLAUDE.md) points
to; read it rather than guessing. What does *not* carry over: the changelog here is the
GitHub releases page (no towncrier fragments), there are no lazy `__init__.pyi` stubs, and
tests use the small `tests/test_raw.fif` instead of the MNE testing dataset.

That also means MNE-Python's
[policy on AI assistance](https://github.com/mne-tools/mne-python/blob/main/CONTRIBUTING.md#policy-on-ai-assistance-in-contributions)
applies: fully-automated submissions are not accepted, every change must be reviewed and
understood by the human submitting it, and AI use must be disclosed in the PR description.

## Layout

- `_pg_figure.py` — `MNEQtBrowser(BrowserBase, QMainWindow)`: builds the plot, wires
  signals, toolbar/keyboard actions, annotation glue, and the `_fake_click`,
  `_fake_keypress`, `_fake_scroll` helpers that the tests drive it with.
- `_graphic_items.py` — items *inside* the plot: `DataTrace`, `AnnotRegion`,
  `SingleChannelAnnot`, scale bars, `VLine`, `EventLine`, `Crosshair`. Drawing order is
  set by the `_Z_*` constants at the top of the file.
- `_widgets.py` — Qt widgets *around* the plot: `BrowserView`, `RawViewBox`,
  `ChannelAxis`/`TimeAxis`, `OverviewBar`, scrollbars, `AnnotationDock`.
- `_dialogs.py` — settings, help, projectors, channel selection, annotation editing.
- `_colors.py` — oklab conversions and `_get_color` (theme-aware, cached).
- `_utils.py` — `_disconnect`, `_methpartial`, `_q_font`, physical-unit conversions.
- `figure.py` / `__init__.py` — import shim and the `_browser_instances` list that keeps
  figures from being garbage-collected out from under Qt.

## Things that bite

- `self.mne` is MNE-Python's shared parameter namespace, not ours alone: `t_start`,
  `duration`, `ch_start`, `traces`, `regions`, `viewbox`, `plt`, … Anything `BrowserBase`
  reads or writes must live there under the name it expects.
- The authoritative view range is `self.mne.viewbox.viewRange()`; `mne.t_start` and
  `mne.duration` are caches refreshed in `_xrange_changed`.
- Redraw path: `plt.setXRange()` → `sigXRangeChanged` → `_xrange_changed` →
  `_redraw(update_data=True)` plus scale bars, overview bar, annotations.
- **A resize emits no range signal.** Anything positioned in pixel units (text rows, label
  offsets) needs `viewbox.sigResized` too, or it silently goes stale until the next scroll.
- Qt geometry read before layout is stale — e.g. `TextItem.boundingRect()` returns empty
  bounds until the item has been painted. Prefer `QFontMetrics` on the font you set.
- Signal connections that outlive the figure keep it alive; `closeEvent` `_disconnect`s
  them explicitly. Prefer one figure-level connection plus a loop over per-item
  connections that nothing ever tears down.

## Tests

`pytest tests/` — `tests/test_pg_specific.py` for behavior, `tests/test_speed.py` for
scroll benchmarks. `pg_backend` comes from `mne.conftest`; `raw_orig` is session-scoped,
so `.copy()` before mutating it. `fig.test_mode = True` makes message boxes non-modal.
Warnings are errors. Run pre-commit (ruff) before committing.

If the package is pip-installed editable from a different checkout, prefix commands with
`PYTHONPATH=$PWD/src` so the working tree is what gets imported.

## Look at it before you believe it

This is a GUI: a change can pass every assertion and still be wrong on screen. Before
calling a visual change done, render a representative example and actually look at the
image, *and* print the numbers behind it — the two catch different bugs.

```python
fig = raw.plot(duration=10, show=False)
fig.test_mode = True
fig.resize(900, 450)
fig.show()
app = QApplication.instance()
fig.mne.plt.setXRange(t, t + 10, padding=0)
app.processEvents()          # required, or you grab the pre-layout frame
fig.grab().save("shot.png")  # then read the PNG
print({r.description: r.label_item.pos() for r in fig.mne.regions})
```

From gh-210, which produced this file:

- Stacked annotation labels rendered *underneath* the traces (wrong z-value) and were
  unreadable over dense data. Both were obvious in the screenshot and invisible to tests.
- The label rows were unevenly spaced because of a stale `boundingRect()`. That looked
  fine in the screenshot and was obvious the moment the y-positions were printed.

Practical notes: use a *representative* example, not a minimal one — six channels of
full-scale noise exposes legibility and overlap problems that clean sample data hides.
Cover several states in one script (a couple of scroll positions, a resize, both themes).
Keep these scripts in scratch space, and promote whatever they caught into a real test.
