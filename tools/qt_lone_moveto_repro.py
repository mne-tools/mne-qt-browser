"""Standalone Qt-only reproducer for the cosmetic stroker's out-of-bounds read.

QCosmeticStroker::drawPath reads points[-2], points[-1] when a path starts with a lone
MoveTo element, i.e. a MoveTo immediately followed by another MoveTo.
QPainterPath.moveTo() collapses consecutive moveTos, so the path is built through
QDataStream, the way pyqtgraph's arrayToQPath does it for curves with non-finite
samples. Run under Guard Malloc with underrun protection to make the read fatal every
time::

    DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib MALLOC_PROTECT_BEFORE=1 \
        python tools/qt_lone_moveto_repro.py
"""

import struct
import sys

from PySide6.QtCore import QByteArray, QDataStream, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter, QPainterPath, QPen

app = QGuiApplication(sys.argv)

# (connect flag, x, y): 0 = MoveTo, 1 = LineTo -- serialized QPainterPath format
# More than 128 points, so QVectorPath's point array is heap allocated rather than
# living in QVarLengthArray's inline buffer (only heap memory can have an unmapped
# page before it)
verts = [(0, 10.0, 10.0), (0, 20.0, 20.0)] + [
    (1, 20.0 + i * 0.3, 20.0 + (i % 7)) for i in range(200)
]
buf = struct.pack(">i", len(verts))
for c, x, y in verts:
    buf += struct.pack(">idd", c, x, y)
buf += struct.pack(">ii", 0, 0)  # cStart, fillRule
path = QPainterPath()
QDataStream(QByteArray(buf)) >> path
print(
    "elements:",
    path.elementCount(),
    [
        (path.elementAt(i).type.name, path.elementAt(i).x, path.elementAt(i).y)
        for i in range(3)
    ],
)

img = QImage(100, 100, QImage.Format.Format_ARGB32_Premultiplied)
img.fill(Qt.GlobalColor.white)
p = QPainter(img)
pen = QPen(Qt.GlobalColor.black, 1)
pen.setCosmetic(
    True
)  # cosmetic width<=1 pen -> QRasterPaintEngine fast_pen -> QCosmeticStroker
p.setPen(pen)
p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
p.drawPath(path)
p.end()
print("drawn OK")
