# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## What this is

mne-qt-browser is the PyQtGraph-based backend for MNE-Python's raw/epochs/ICA browser
(`raw.plot()` and friends once `mne.viz.set_browser_backend("qt")` is active). MNE-Python
owns the data and the browser API (`mne.viz._figure.BrowserBase`); this package owns
everything you see.

## Follow MNE-Python's conventions

This package is a subsidiary of MNE-Python. Unless something below says otherwise, follow
[MNE-Python's AGENTS.md](https://github.com/mne-tools/mne-python/blob/main/AGENTS.md)
(read it rather than guessing): naming, numpydoc style with its local deviations, absolute
and lazily-nested imports, deprecation policy, compact tests, license rules for adapted
code, and in particular its
[policy on AI assistance](https://github.com/mne-tools/mne-python/blob/main/CONTRIBUTING.md#policy-on-ai-assistance-in-contributions):

- Work test-first: write (or extend) a test that fails for the right reason, then make it
  pass. Promote anything a throwaway script caught into a real test.
- Do not open pull requests, push, or commit unless explicitly asked; the human submitting
  the change must review, understand, and disclose AI use in the PR description.
- Keep changes minimal and scoped to the request; mention, don't silently fix, unrelated
  problems you notice.

What does *not* carry over from MNE-Python:

- The changelog is the GitHub releases page, so there are no towncrier fragments to add.
- There are no lazy `__init__.pyi` stubs; the public API is just `mne_qt_browser.figure`.
- Tests use the small bundled `tests/test_raw.fif`, not the MNE testing dataset.

## Verify before you trust this file

Nothing keeps this file in sync with the code, and much of what it names is not even ours:
`self.mne`'s attributes, the order `BrowserBase` calls our methods in, and the `pg_backend`
fixture are MNE-Python private API that moves in a different repo on its own schedule, and
the Qt/pyqtgraph behaviors described below are empirical rather than documented. What is
written here is meant to be the *durable* shape of the package, but specific names,
signals, and the snippets in the last section do drift. So treat this file as a map of
where to look and what to watch out for, not as an API reference: grep for the name, read
the code around it, and confirm the behavior by actually running it before you rely on it.

If something here turns out to be stale, update this file as part of your change and
mention it. If the drift is real but outside the scope of what you were asked to do, say so
in your summary rather than quietly working around it.

## Layout

- `src/mne_qt_browser/_pg_figure.py` — `MNEQtBrowser(BrowserBase, QMainWindow)`: builds the
  plot, wires signals, toolbar/keyboard actions, annotation glue, and the `_fake_click`,
  `_fake_keypress`, `_fake_scroll` helpers that the tests drive it with.
- `_graphic_items.py` — items *inside* the plot: `DataTrace`, `AnnotRegion`,
  `SingleChannelAnnot`, scale bars, `VLine`, `EventLine`, `Crosshair`. Drawing order is
  set by the `_Z_*` constants at the top of the file.
- `_widgets.py` — Qt widgets *around* the plot: `BrowserView`, `RawViewBox`,
  `ChannelAxis`/`TimeAxis`, `OverviewBar`, scrollbars, `AnnotationDock`.
- `_dialogs.py` — settings, help, projectors, channel selection, annotation editing.
- `_colors.py` — OkLab conversions and `_get_color` (theme-aware, cached).
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
  `_redraw(update_data=True)` plus scale bars, overview bar, annotations. Everything on
  that path runs on every scroll, so keep per-item work there cheap.
- **A resize emits no range signal.** Anything positioned in pixel units (text rows, label
  offsets) needs `viewbox.sigResized` too, or it silently goes stale until the next scroll.
- Qt geometry read before layout is stale — e.g. `TextItem.boundingRect()` returns empty
  bounds until the item has been painted. Prefer `QFontMetrics` on the font you set.
- Signal connections that outlive the figure keep it alive, and leftover references cause
  garbage-collection segfaults at teardown (often only on CI). `closeEvent` `_disconnect`s
  our connections and `delattr`s Qt objects on `self.mne` explicitly: add any new Qt object
  stored on `self.mne` to that list. Prefer one figure-level connection plus a loop over
  per-item connections that nothing ever tears down, and disconnect *our* slot rather than
  blanket-disconnecting a pyqtgraph signal that pyqtgraph itself listens to (e.g.
  `viewbox.sigResized`).
- Colors are stored for light mode and converted for dark mode via `_get_color(color,
  self.mne.dark)`; `mne.dark` comes from the Qt palette, so `theme="dark"` alone does not
  make it `True` under an offscreen/xvfb platform.

## Tests

`pytest tests/` — `tests/test_pg_specific.py` for behavior, `tests/test_speed.py` for
scroll benchmarks. `pg_backend` comes from `mne.conftest`; `raw_orig` is session-scoped,
so `.copy()` before mutating it. `fig.test_mode = True` makes message boxes non-modal.
Warnings are errors. Run `pre-commit run --all-files` before handing work back.

Known (fixed) crash: painting a trace whose *first* sample is non-finite used to SIGBUS
sporadically on the macOS CI runners (`Fatal Python error: Bus error` inside
`QPainter.drawPath`, exit code 138, no `FAILED` line). pyqtgraph turns a leading NaN into
a lone `MoveTo` element, and `QCosmeticStroker::drawPath` (every Qt 5/6) treats such a
subpath as closed and reads the two points before it, i.e. before the point array.
`DataTrace.update_data` now drops leading non-finite samples on the `connect="finite"`
path (`test_no_lone_leading_moveto`); `tools/qt_lone_moveto_repro.py` is the Qt-only
reproducer. Such out-of-bounds reads are made deterministic on macOS with
`DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib MALLOC_PROTECT_BEFORE=1`, and the
workflow supports SSH debugging (`[actions ssh]` in the PR title or commit message); on a
runner, loop the test under `lldb --batch -o "settings set target.process.stop-on-exec
false" -o run -k "bt 40" -k "register read" -- $(which python) -m pytest ...` to catch it.

Headless: run GUI tests and scripts with `xvfb-run -a` (or `QT_QPA_PLATFORM=offscreen`).
Prefer driving the figure through `pytest-qt`/`_fake_*` helpers over OS-level input
injection. If the package is pip-installed editable from a *different* checkout (e.g. you
are in a git worktree), prefix commands with `PYTHONPATH=$PWD/src` so the working tree is
what gets imported — check with `python -c "import mne_qt_browser; print(mne_qt_browser.__file__)"`.

## Look at it before you believe it

This is a GUI: a change can pass every assertion and still be wrong on screen. Before
calling a visual change done, render a representative example and actually look at the
image, *and* print the numbers behind it — the two catch different bugs (a wrong z-value
is obvious in a screenshot and invisible to tests; uneven label spacing from a stale
`boundingRect()` looks fine in a screenshot and is obvious once the positions are printed).

```python
fig = raw.plot(duration=10, show=False)
fig.test_mode = True
fig.resize(900, 450)
fig.show()
app = QApplication.instance()
fig.mne.plt.setXRange(t, t + 10, padding=0)
app.processEvents()  # required, or you grab the pre-layout frame
fig.grab().save("shot.png")  # then read the PNG
print({r.description: r.label_item.pos() for r in fig.mne.regions})
```

Use a *representative* example, not a minimal one — a few channels of full-scale noise
expose legibility and overlap problems that clean sample data hides. Cover several states in
one script (a couple of scroll positions, a resize, both themes, butterfly mode). Keep these
scripts in scratch space, and promote whatever they caught into a real test.
