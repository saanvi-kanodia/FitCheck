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

    h, w = img.shape[:2]
    # Keep sizes consistent across analysers: clamp to a range set by the frame's
    # shorter side, so portrait phone videos don't get oversized text.
    short = min(h, w)
    size = int(round(fontScale * _PX_PER_SCALE))
    size = max(int(short * 0.026), min(size, int(short * 0.036)), 10)
    font = _font(size, heavy=thickness >= 2)
    text = str(text)
    x, y = int(org[0]), int(org[1])

    pad = max(4, size // 4)
    left, top, right, bottom = font.getbbox(text, anchor="ls")
    x0, y0 = max(0, x + left - pad), max(0, y + top - pad)
    x1, y1 = min(w, x + right + pad), min(h, y + bottom + pad)
    if x0 >= x1 or y0 >= y1:
        return img

    region = img[y0:y1, x0:x1]
    # Put a dark pill behind the text so it reads on any footage. Skip it where the
    # analyser already drew a plain backing (the push-up bar), i.e. the area is flat.
    if region.std() >= 12 or region.mean() > 90:
        pill = Image.new("L", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(pill).rounded_rectangle((0, 0, x1 - x0 - 1, y1 - y0 - 1), radius=pad + 2, fill=200)
        a = (np.asarray(pill, dtype=np.float32) / 255.0)[:, :, None]
        region[:] = (region * (1 - a) + np.array([18, 16, 14], np.float32) * a).astype(np.uint8)

    # Dark text colours (pure blue/red) are unreadable on a dark backing: lighten them.
    b, g, r = (int(c) for c in color[:3]) if len(color) >= 3 else (255, 255, 255)
    if 0.299 * r + 0.587 * g + 0.114 * b < 150:
        r, g, b = (int(c + (255 - c) * 0.45) for c in (r, g, b))

    patch = Image.fromarray(region[:, :, ::-1])
    ImageDraw.Draw(patch).text((x - x0, y - y0), text, font=font, fill=(r, g, b), anchor="ls")
    img[y0:y1, x0:x1] = np.asarray(patch)[:, :, ::-1]
    return img


def install():
    cv2.putText = put_text
