"""Replace OpenCV's built-in Hershey text with a real font.

cv2.putText can only draw OpenCV's blocky line fonts and cannot draw characters
such as the degree sign. install() swaps it for a version that renders text
with Pillow using a TrueType font, so every analyser gets clean overlay text
without being edited.
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# (path, index for Bold, index for Medium). The first one that exists is used.
_CANDIDATES = [
    ("/System/Library/Fonts/HelveticaNeue.ttc", 1, 10),          # macOS
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0, 0),  # Linux
]

# Hershey fonts at scale 1.0 are roughly this many pixels tall.
_PX_PER_SCALE = 30

_orig_put_text = cv2.putText
_cache = {}


def _font_spec():
    for path, bold, medium in _CANDIDATES:
        if os.path.exists(path):
            return path, bold, medium
    import matplotlib  # bundled with the requirements, so always available
    path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf")
    return path, 0, 0


_SPEC = _font_spec()


def _font(size, heavy):
    key = (size, heavy)
    if key not in _cache:
        path, bold, medium = _SPEC
        _cache[key] = ImageFont.truetype(path, size, index=bold if heavy else medium)
    return _cache[key]


def put_text(img, text, org, fontFace, fontScale, color, thickness=1, lineType=None,
             bottomLeftOrigin=False):
    if (not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3
            or img.shape[2] != 3 or bottomLeftOrigin):
        return _orig_put_text(img, text, org, fontFace, fontScale, color, thickness,
                              lineType or cv2.LINE_8, bottomLeftOrigin)

    size = max(8, int(round(fontScale * _PX_PER_SCALE)))
    font = _font(size, heavy=thickness >= 2)
    x, y = int(org[0]), int(org[1])

    # Draw only on the patch of the frame the text covers, which keeps it fast.
    left, top, right, bottom = font.getbbox(str(text), anchor="ls")
    h, w = img.shape[:2]
    x0, y0 = max(0, x + left - 2), max(0, y + top - 2)
    x1, y1 = min(w, x + right + 2), min(h, y + bottom + 2)
    if x0 >= x1 or y0 >= y1:
        return img

    patch = Image.fromarray(img[y0:y1, x0:x1][:, :, ::-1])
    rgb = tuple(int(c) for c in (color[2], color[1], color[0])) if len(color) >= 3 else (255, 255, 255)
    ImageDraw.Draw(patch).text((x - x0, y - y0), str(text), font=font, fill=rgb, anchor="ls")
    img[y0:y1, x0:x1] = np.asarray(patch)[:, :, ::-1]
    return img


def install():
    cv2.putText = put_text
