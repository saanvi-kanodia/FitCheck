"""Run with: .venv/bin/python -m unittest discover -s tests/unit"""
import math
import os
import sys
import tempfile
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysers"))

from core import pose  # noqa: E402


def landmarks(width, height, visibility=1.0, **pixels):
    """A full set of 33 landmarks with the named ones placed at the given pixel positions."""
    rows = np.zeros((len(pose.NAMES), 5), np.float32)
    rows[:, :2], rows[:, 3], rows[:, 4] = 0.5, visibility, np.nan
    for name, (x, y) in pixels.items():
        rows[pose.INDEX[name], :2] = x / width, y / height
    return rows


class BodyPoints(unittest.TestCase):
    def test_angle_is_true_in_portrait_and_landscape(self):
        ankle = (300 + 100 * math.sin(math.radians(60)), 200 + 100 * math.cos(math.radians(60)))
        for width, height in ((640, 360), (360, 640)):
            body = pose.Body(landmarks(width, height, left_hip=(300, 100), left_knee=(300, 200), left_ankle=ankle),
                             width, height, side="left")
            self.assertAlmostEqual(body.angle("hip", "knee", "ankle"), 120, places=3)

    def test_points_come_back_in_pixels(self):
        body = pose.Body(landmarks(640, 360, nose=(64, 36)), 640, 360)
        x, y = body.point("nose")
        self.assertAlmostEqual(x, 64, places=3)
        self.assertAlmostEqual(y, 36, places=3)

    def test_a_joint_without_a_side_means_the_side_facing_the_camera(self):
        rows = landmarks(640, 360, left_knee=(100, 100), right_knee=(500, 300))
        self.assertAlmostEqual(pose.Body(rows, 640, 360, side="left").point("knee")[0], 100, places=3)
        self.assertAlmostEqual(pose.Body(rows, 640, 360, side="right").point("knee")[0], 500, places=3)
        self.assertAlmostEqual(pose.Body(rows, 640, 360, side="right").point("knee", side="left")[0], 100, places=3)

    def test_seen_needs_a_confident_point_inside_the_frame(self):
        rows = landmarks(640, 360, left_knee=(100, 100), left_ankle=(100, 400), left_hip=(100, 50))
        rows[pose.INDEX["left_hip"], 3] = 0.3                      # MediaPipe is only guessing at the hip
        body = pose.Body(rows, 640, 360, side="left")
        self.assertTrue(body.seen("knee"))
        self.assertFalse(body.seen("ankle"))                       # below the bottom edge
        self.assertFalse(body.seen("hip"))
        self.assertFalse(body.seen("knee", "hip"))                 # every joint asked for must be seen


class Sides(unittest.TestCase):
    def rows(self, left, right):
        r = landmarks(640, 360)
        r[pose._LEFT, 3], r[pose._RIGHT, 3] = left, right
        return r

    def test_picks_the_side_the_camera_can_see(self):
        self.assertEqual(pose.SideChooser().update(self.rows(0.95, 0.4)), "left")
        self.assertEqual(pose.SideChooser().update(self.rows(0.4, 0.95)), "right")

    def test_a_few_bad_frames_do_not_flip_it(self):
        chooser = pose.SideChooser()
        for _ in range(30):
            chooser.update(self.rows(0.95, 0.4))
        for _ in range(3):                                          # a brief glitch
            side = chooser.update(self.rows(0.3, 0.9))
        self.assertEqual(side, "left")

    def test_follows_the_person_when_they_turn_round(self):
        chooser = pose.SideChooser()
        for _ in range(30):
            chooser.update(self.rows(0.95, 0.4))
        for _ in range(30):
            side = chooser.update(self.rows(0.4, 0.95))
        self.assertEqual(side, "right")

    def test_stays_put_when_both_sides_are_equally_visible(self):
        chooser = pose.SideChooser()
        first = chooser.update(self.rows(0.9, 0.9))
        for _ in range(60):
            side = chooser.update(self.rows(0.9, 0.92))
        self.assertEqual(side, first)


def person(turn, lying=False, width=640, height=360, scale=300.0):
    """Landmarks of a simple torso turned `turn` degrees away from facing the camera.

    Returns (image rows, world rows): shoulders 36 cm apart, hips 24 cm, torso 50 cm long.
    lying=True puts the torso along the ground, as in a plank filmed from the side.
    """
    image, world = landmarks(width, height), np.zeros((len(pose.NAMES), 4), np.float32)
    t = math.radians(turn)
    for name, across, along in (("left_shoulder", 0.18, 0.0), ("right_shoulder", -0.18, 0.0),
                                ("left_hip", 0.12, 0.5), ("right_hip", -0.12, 0.5)):
        x, z = across * math.cos(t), across * math.sin(t)       # the body's left-right line, turned
        wx, wy = (along, x) if lying else (x, along)             # lying: the torso runs along the picture
        world[pose.INDEX[name], :3] = wx, wy, z
        image[pose.INDEX[name], :2] = (width / 2 + wx * scale) / width, (height / 4 + wy * scale) / height
    return image, world


class CameraPosition(unittest.TestCase):
    def test_turn_and_width_for_known_positions(self):
        for turn, width in ((0, 0.6), (45, 0.424), (90, 0.0)):
            image, world = person(turn)
            got_turn, got_width = pose.facing(image, world, 640, 360)
            self.assertAlmostEqual(got_turn, turn, places=3)
            self.assertAlmostEqual(got_width, width, places=2)

    def test_the_frame_shape_does_not_change_the_answer(self):
        landscape = pose.facing(*person(30, width=640, height=360), 640, 360)
        portrait = pose.facing(*person(30, width=360, height=640), 360, 640)
        self.assertAlmostEqual(landscape[0], portrait[0], places=3)
        self.assertAlmostEqual(landscape[1], portrait[1], places=3)

    def test_someone_lying_side_on_reads_as_side_on(self):
        turn, width = pose.facing(*person(88, lying=True), 640, 360)
        self.assertAlmostEqual(turn, 88, places=3)
        self.assertLess(width, 0.05)

    def test_view_names_only_clear_cases(self):
        frames = lambda turn: [pose.facing(*person(turn), 640, 360)] * 9
        self.assertEqual(pose.view(frames(5)), "front")
        self.assertEqual(pose.view(frames(85)), "side")
        self.assertIsNone(pose.view(frames(45)))                   # a diagonal view is left alone
        self.assertIsNone(pose.view([]))

    def test_view_is_none_when_the_two_measures_disagree(self):
        self.assertIsNone(pose.view([(5.0, 0.05)] * 9))           # 3D says front, the picture says side
        self.assertIsNone(pose.view([(85.0, 0.6)] * 9))

    def test_view_follows_the_bulk_of_the_video(self):
        side, front = pose.facing(*person(85), 640, 360), pose.facing(*person(5), 640, 360)
        self.assertEqual(pose.view([side] * 20 + [front] * 4), "side")


class FakeModel:
    """Stands in for MediaPipe: hands back prepared landmark rows, one frame per call."""

    def __init__(self, frames):
        self.frames, self.calls = list(frames), 0

    def process(self, _rgb):
        from mediapipe.framework.formats import landmark_pb2
        rows = self.frames[self.calls]
        self.calls += 1
        if rows is None:
            return types.SimpleNamespace(pose_landmarks=None, pose_world_landmarks=None)
        return types.SimpleNamespace(
            pose_landmarks=pose._to_landmarks(rows, landmark_pb2.NormalizedLandmarkList),
            pose_world_landmarks=pose._to_landmarks(rows[:, :4], landmark_pb2.LandmarkList))

    def close(self):
        pass


class Cache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.video = os.path.join(self.tmp.name, "clip.mp4")
        with open(self.video, "wb") as f:
            f.write(b"not really a video, but the cache only needs its bytes")
        self.frame = np.zeros((360, 640, 3), np.uint8)
        rng = np.random.default_rng(1)
        self.frames = [landmarks(640, 360), None, landmarks(640, 360)]
        for rows in (self.frames[0], self.frames[2]):
            rows[:, :4] = rng.random((len(pose.NAMES), 4), dtype=np.float32)
        self.frames[2][5, 4] = 0.75                                 # one point also has a presence score
        os.environ["FITCHECK_POSE_CACHE"] = os.path.join(self.tmp.name, "cache")

    def tearDown(self):
        os.environ.pop("FITCHECK_POSE_CACHE", None)
        self.tmp.cleanup()

    def run_live(self, **options):
        tracker = pose.Tracker(video=self.video, **options)
        tracker._model = FakeModel(self.frames)
        out = [tracker.process(self.frame) for _ in self.frames]
        tracker.close()
        return out

    def test_a_second_run_replays_the_first_exactly_without_the_model(self):
        live = self.run_live()
        again = pose.Tracker(video=self.video)
        for first in live:
            second = again.process(self.frame)
            if first.pose_landmarks is None:
                self.assertIsNone(second.pose_landmarks)
                self.assertIsNone(again.body)
                continue
            self.assertEqual(first.pose_landmarks, second.pose_landmarks)          # every field, bit for bit
            self.assertEqual(first.pose_world_landmarks, second.pose_world_landmarks)
        self.assertIsNone(again._model)                                            # MediaPipe was never loaded
        with self.assertRaises(RuntimeError):                                      # no silent guessing past the end
            again.process(self.frame)

    def test_other_settings_or_another_video_do_not_reuse_it(self):
        self.run_live()
        self.assertIsNotNone(pose.Tracker(video=self.video)._saved)
        self.assertIsNone(pose.Tracker(video=self.video, model_complexity=0)._saved)
        other = os.path.join(self.tmp.name, "other.mp4")
        with open(other, "wb") as f:
            f.write(b"a different video")
        self.assertIsNone(pose.Tracker(video=other)._saved)

    def test_nothing_is_written_unless_a_cache_folder_is_named(self):
        os.environ.pop("FITCHECK_POSE_CACHE")
        self.run_live()
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "cache")))

    def test_save_all_writes_trackers_that_were_never_closed(self):
        tracker = pose.Tracker(video=self.video)
        tracker._model = FakeModel(self.frames)
        for _ in self.frames:
            tracker.process(self.frame)
        pose.save_all()                                             # what webapp/runner.py does on the way out
        self.assertIsNotNone(pose.Tracker(video=self.video)._saved)


if __name__ == "__main__":
    unittest.main()
