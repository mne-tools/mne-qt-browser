"""Color utility functions."""

# License: BSD-3-Clause
# Copyright the MNE Qt Browser contributors.

import functools

import numpy as np
from mne.utils import _to_rgb, logger
from pyqtgraph import mkColor
from qtpy.QtGui import QColor

_REF_XYZ = (95.047, 100.000, 108.883)  # D65, 2 deg observer

# Mostly chosen from https://matplotlib.org/3.1.0/gallery/color/named_colors.html
_dark_dict = {
    # 'w' (bgcolor)
    (255, 255, 255): (30, 30, 30),  # Safari's centered info panel background
    # 'k' (eeg, eog, emg, misc, stim, resp, chpi, exci, ias, syst, dipole, gof, ...)
    (0, 0, 0): (255, 255, 255),  # 'w'
    # 'darkblue' (mag)
    (0, 0, 139): (173, 216, 230),  # 'lightblue'
    # 'b' (grad, hbr)
    (0, 0, 255): (100, 149, 237),  # 'cornflowerblue'
    # 'steelblue' (ref_meg)
    (70, 130, 180): (176, 196, 222),  # 'lightsteelblue'
    # 'm' (ecg)
    (191, 0, 191): (238, 130, 238),  # 'violet'
    # 'saddlebrown' (seeg)
    (139, 69, 19): (244, 164, 96),  # 'sandybrown'
    # 'seagreen' (dbs)
    (46, 139, 87): (32, 178, 170),  # 'lightseagreen'
    # '#AA3377' (hbo), closest to 'mediumvioletred'
    (170, 51, 119): (255, 105, 180),  # 'hotpink'
    # 'lightgray' (bad_color)
    (211, 211, 211): (105, 105, 105),  # 'dimgray'
    # 'cyan' (event_color)
    (0, 255, 255): (0, 139, 139),  # 'darkcyan'
}


def _rgb_to_lab(rgb):
    # Alternatives include skimage (too heavy, compiled and requires networkx),
    # PIL.ImageCms (not CIElab, didn't seem to work very well), or
    # colorspacious (unmaintained), so we'll just go based on the reference
    # https://www.easyrgb.com/en/math.php

    # sRGB -> XYZ
    xyz = np.array(rgb, float)  # make a copy
    assert xyz.shape == (3,), rgb.shape
    mask = xyz > 0.04045
    xyz[mask] = ((xyz[mask] + 0.055) / 1.055) ** 2.4
    xyz[~mask] /= 12.92
    xyz *= 100
    xyz = [
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
    ] @ xyz

    # XYZ -> CIELab
    # skimage.color defaults to lluminant='D65', observer='2' so we'll use those same
    # values
    lab = xyz / _REF_XYZ

    mask = lab > 0.008856
    lab[mask] **= 1 / 3
    lab[~mask] = 7.787 * lab[~mask] + 16 / 116
    lab = np.array(
        [
            (116 * lab[1]) - 16,
            500 * (lab[0] - lab[1]),
            200 * (lab[1] - lab[2]),
        ]
    )

    return lab


def _lab_to_rgb(lab):
    lab = np.array(lab, float)  # make a copy
    assert lab.shape == (3,), lab.shape

    # CIELab -> XYZ
    y = (lab[0] + 16) / 116
    x = lab[1] / 500 + y
    z = y - lab[2] / 200
    xyz = np.array([x, y, z], float)

    mask = xyz > 0.20689303442296383  # same as xyz ** 3 > 0.008856
    xyz[mask] **= 3
    xyz[~mask] = (xyz[~mask] - 16 / 116) / 7.787

    xyz *= _REF_XYZ

    # XYZ -> sRGB
    rgb = [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ] @ (xyz / 100)

    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * (rgb[mask] ** (1 / 2.4)) - 0.055
    rgb[~mask] *= 12.92
    np.clip(rgb, 0, 1, out=rgb)

    return rgb


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
            lab = _rgb_to_lab(rgba[:3])
            lab[0] = 100.0 - lab[0]
            rgba[:3] = _lab_to_rgb(lab)
            color.setRgbF(*rgba)

    return color
