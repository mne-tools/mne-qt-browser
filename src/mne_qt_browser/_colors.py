"""Color utility functions."""

# License: BSD-3-Clause
# Copyright the MNE Qt Browser contributors.

import functools

import numpy as np
from mne.utils import _to_rgb, logger
from pyqtgraph import mkColor
from qtpy.QtGui import QColor

# Mostly chosen from https://matplotlib.org/3.1.0/gallery/color/named_colors.html
_dark_dict = {
    # 'w' (bgcolor)
    (255, 255, 255): (30, 30, 30),  # Safari's centered info panel background
    # 'k' (eeg, eog, emg, misc, stim, resp, chpi, exci, ias, syst, dipole, gof, ...)
    (0, 0, 0): (255, 255, 255),  # 'w'
    # 'darkblue' (mag) and 'b' (grad, hbr) are absent on purpose: the inversion below
    # handles them with no hue error, where no named pair stayed distinguishable
    # 'steelblue' (ref_meg), a midtone that would invert to only 1.9:1 contrast
    (70, 130, 180): (176, 196, 222),  # 'lightsteelblue'
    # 'm' (ecg)
    (191, 0, 191): (238, 130, 238),  # 'violet'
    # 'saddlebrown' (seeg)
    (139, 69, 19): (244, 164, 96),  # 'sandybrown'
    # 'seagreen' (dbs)
    (46, 139, 87): (60, 179, 113),  # 'mediumseagreen' (same oklch hue as seagreen)
    # '#AA3377' (hbo), closest to 'mediumvioletred'
    (170, 51, 119): (255, 105, 180),  # 'hotpink'
    # 'lightgray' (bad_color)
    (211, 211, 211): (105, 105, 105),  # 'dimgray'
    # 'cyan' (event_color)
    (0, 255, 255): (0, 139, 139),  # 'darkcyan'
}


# Oklab is more perceptually uniform than CIELab, especially in hue. Matrices adapted
# from https://bottosson.github.io/posts/oklab/ (public domain, or MIT if preferred).
_M_SRGB_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_M_LMS_TO_OKLAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
# Invert numerically: the published inverses are rounded separately, so hard-coding
# them would make round-trips inexact
_M_OKLAB_TO_LMS = np.linalg.inv(_M_LMS_TO_OKLAB)
_M_LMS_TO_SRGB = np.linalg.inv(_M_SRGB_TO_LMS)


def _rgb_to_oklab(rgb):
    rgb = np.array(rgb, float)  # make a copy
    assert rgb.shape == (3,), rgb.shape

    # sRGB -> linear sRGB
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] /= 12.92

    # linear sRGB -> Oklab
    return _M_LMS_TO_OKLAB @ np.cbrt(_M_SRGB_TO_LMS @ rgb)


def _oklab_to_linear_srgb(lab):
    return _M_LMS_TO_SRGB @ (_M_OKLAB_TO_LMS @ lab) ** 3


def _oklab_to_rgb(lab):
    lab = np.array(lab, float)  # make a copy
    assert lab.shape == (3,), lab.shape

    rgb = _oklab_to_linear_srgb(lab)
    eps = 1e-6
    if rgb.min() < -eps or rgb.max() > 1 + eps:
        # Out of gamut: clipping RGB would shift hue and lightness, so instead hold
        # those fixed and search for the largest in-gamut chroma
        lab[0] = np.clip(lab[0], 0, 1)  # gray axis is in gamut for 0 <= L <= 1
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2
            test = _oklab_to_linear_srgb(lab * [1.0, mid, mid])
            if test.min() < -eps or test.max() > 1 + eps:
                hi = mid
            else:
                lo = mid
        rgb = _oklab_to_linear_srgb(lab * [1.0, lo, lo])
    np.clip(rgb, 0, 1, out=rgb)

    # linear sRGB -> sRGB
    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * (rgb[mask] ** (1 / 2.4)) - 0.055
    rgb[~mask] *= 12.92

    return rgb


# Mirroring alone leaves midtones too close to the background, so clamp to a floor:
# 0.56 is the lowest oklab lightness clearing 3:1 (WCAG, graphical objects) at every
# hue. A floor rather than a rescaling, so colors above it keep their spacing and ones
# differing only in lightness (e.g. 'darkblue' and 'b') stay distinguishable
_MIN_LIGHTNESS = 0.56


def _get_color(color_spec, invert=False):
    """Wrap mkColor to accept all possible Matplotlib color specifiers."""
    if isinstance(color_spec, np.ndarray):
        color_spec = tuple(color_spec)
    # We have to pass to QColor here to make a copy, because we want to be able to call
    # .setAlpha(...) etc. on it, and this would otherwise affect the cache.
    return QColor(_get_color_cached(color_spec=color_spec, invert=invert))


@functools.lru_cache(maxsize=100)
def _get_color_cached(*, color_spec, invert):
    orig_spec = color_spec
    try:
        # Convert matplotlib color names if possible
        color_spec = _to_rgb(color_spec, alpha=True)
    except ValueError:
        pass

    # Convert tuples of floats from 0-1 to 0-255 for PyQtGraph
    if isinstance(color_spec, tuple) and all([i <= 1 for i in color_spec]):
        color_spec = tuple([int(i * 255) for i in color_spec])

    try:
        color = mkColor(color_spec)
    except ValueError:
        raise ValueError(
            f'"{color_spec}" is not a valid Matplotlib color specifier!'
        ) from None
    if invert:
        # First see if the color is in our inversion dictionary
        key = color.getRgb()
        assert len(key) == 4
        if key[:3] in _dark_dict:
            color.setRgb(*(_dark_dict[key[:3]] + key[-1:]))
        else:
            logger.debug(f"Missed {key} from {orig_spec}")
            rgba = np.array(color.getRgbF())
            lab = _rgb_to_oklab(rgba[:3])
            # Mirror the lightness, keeping hue and chroma (a, b) intact
            lab[0] = max(1.0 - lab[0], _MIN_LIGHTNESS)
            rgba[:3] = _oklab_to_rgb(lab)
            color.setRgbF(*rgba)

    return color
