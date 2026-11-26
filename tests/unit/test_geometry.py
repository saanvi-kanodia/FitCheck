"""Run with: .venv/bin/python -m unittest discover -s tests/unit"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysers"))

from core import geometry  # noqa: E402


def old_angle(a, b, c):
    """The angle as the analysers used to work it out, from fractions of the frame."""
    r = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    d = abs(math.degrees(r))
    return 360 - d if d > 180 else d


class Angle(unittest.TestCase):
    def test_known_angles(self):
        self.assertAlmostEqual(geometry.angle((0, 1), (0, 0), (1, 0)), 90)
        self.assertAlmostEqual(geometry.angle((-1, 0), (0, 0), (1, 0)), 180)
        self.assertAlmostEqual(geometry.angle((2, 0), (0, 0), (1, 0)), 0)
        self.assertAlmostEqual(geometry.angle((1, 0), (0, 0), (1, 1)), 45)
        self.assertAlmostEqual(geometry.angle((1, 0), (0, 0), (-1, 1)), 135)

    def test_does_not_depend_on_which_way_the_body_faces(self):
        hip, knee, ankle = (100, 100), (100, 200), (150, 280)
        mirrored = [(640 - x, y) for x, y in (hip, knee, ankle)]
        self.assertAlmostEqual(geometry.angle(hip, knee, ankle), geometry.angle(*mirrored))

    def test_same_bend_reads_the_same_in_portrait_and_landscape(self):
        # One knee bent to a true 120 degrees, at the same pixels in a landscape and a portrait frame.
        hip, knee = (300.0, 100.0), (300.0, 200.0)
        ankle = (knee[0] + 100 * math.sin(math.radians(60)), knee[1] + 100 * math.cos(math.radians(60)))
        self.assertAlmostEqual(geometry.angle(hip, knee, ankle), 120)
        seen = {}
        for name, (w, h) in {"landscape": (640, 360), "portrait": (360, 640)}.items():
            as_fractions = [(x / w, y / h) for x, y in (hip, knee, ankle)]
            seen[name] = old_angle(*as_fractions)
            back_in_pixels = [(x * w, y * h) for x, y in as_fractions]
            self.assertAlmostEqual(geometry.angle(*back_in_pixels), 120)
        # This is the fault being fixed: from fractions, the same knee read about 136 and 108 degrees.
        self.assertAlmostEqual(seen["landscape"], 135.7, delta=0.5)
        self.assertAlmostEqual(seen["portrait"], 108.0, delta=0.5)

    def test_zero_length_limb_does_not_crash(self):
        self.assertEqual(geometry.angle((5, 5), (5, 5), (9, 9)), 0)


if __name__ == "__main__":
    unittest.main()
