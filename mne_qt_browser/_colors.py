"""Color utility functions."""

# License: BSD-3-Clause
# Copyright the MNE-Qt_browser contributors.

import numpy as np

_REF_XYZ = (95.047, 100.000, 108.883)  # D65, 2 deg observer


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
