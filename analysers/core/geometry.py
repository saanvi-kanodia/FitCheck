"""Geometry on body points.

Points are (x, y) in pixels. MediaPipe reports positions as fractions of the frame's width and
height; an angle worked out from those fractions is only right when the frame is square, and
is off by a different amount in portrait and landscape video. In pixels it is a true angle.
"""
import math


def angle(a, b, c):
    """Angle at b between the lines b->a and b->c, in degrees from 0 (folded) to 180 (straight)."""
    ax, ay, cx, cy = a[0] - b[0], a[1] - b[1], c[0] - b[0], c[1] - b[1]
    return abs(math.degrees(math.atan2(ax * cy - ay * cx, ax * cx + ay * cy)))


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
