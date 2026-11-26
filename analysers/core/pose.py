"""Pose tracking shared by every analyser.

`Tracker` stands in for MediaPipe's `Pose`: the same `.process(rgb_image)` call, returning the
same kind of result, so an analyser needs no other change to use it. Around that call it adds
what every analyser needs:

- `tracker.body`, the latest pose as a `Body`: points in pixels (so angles are true angles
  whatever the frame's shape), a check for whether a point can be trusted, and the side of the
  body that faces the camera.
- `facing()` and `view()`: whether a video was filmed from the front or from the side.
- an optional on-disk copy of the raw landmarks, so that re-running an analyser on the same
  video skips the slow pose model. It is off unless FITCHECK_POSE_CACHE names a folder
  (tests/eval/run.py --fast does that).

MediaPipe already steadies its landmarks from frame to frame (a One Euro filter, on by default
through `smooth_landmarks`), so the tracker does not filter them a second time.
"""
import hashlib
import json
import math
import os
import types

import numpy as np

from . import geometry

MIN_VISIBILITY = 0.5   # MediaPipe's "is this point visible" score needed to trust a point
EDGE = 0.02            # a point this far outside the frame (as a fraction of it) still counts as in view

# MediaPipe's 33 pose points, in its order.
NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
)
INDEX = {name: i for i, name in enumerate(NAMES)}
# The joints that decide which side of the body the camera can see.
SIDE_JOINTS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")
_LEFT = [INDEX["left_" + j] for j in SIDE_JOINTS]
_RIGHT = [INDEX["right_" + j] for j in SIDE_JOINTS]

DEFAULTS = dict(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)


class Body:
    """One frame's pose.

    Joints that come in pairs are named without a side ("knee", "shoulder"). By default that
    means the one on the side of the body facing the camera, which is the one MediaPipe can
    really see; pass side="left" or "right" to pick one.
    """

    def __init__(self, landmarks, width, height, side="left"):
        self._lm = landmarks          # rows of x, y (fractions of the frame), z, visibility
        self.width, self.height, self.side = width, height, side

    def _row(self, joint, side):
        return self._lm[INDEX[joint] if joint in INDEX else INDEX[f"{side or self.side}_{joint}"]]

    def point(self, joint, side=None):
        """Where a joint is, as (x, y) in pixels."""
        row = self._row(joint, side)
        return float(row[0]) * self.width, float(row[1]) * self.height

    def visibility(self, joint, side=None):
        return float(self._row(joint, side)[3])

    def seen(self, *joints, side=None):
        """True if every one of these joints is confidently visible and inside the frame."""
        for joint in joints:
            x, y, _, v = self._row(joint, side)[:4]
            if not (v >= MIN_VISIBILITY and -EDGE <= x <= 1 + EDGE and -EDGE <= y <= 1 + EDGE):
                return False
        return True

    def angle(self, a, b, c, side=None):
        """True angle at joint b between joints a and c, in degrees."""
        return geometry.angle(self.point(a, side), self.point(b, side), self.point(c, side))


class SideChooser:
    """Which side of the body faces the camera.

    In a side view MediaPipe sees one arm and leg and guesses the other, and gives the guessed
    one a lower visibility score. The choice follows a running average of the difference and
    only changes once the other side clearly leads, so it does not flip on single frames.
    """
    MARGIN = 0.1      # how far the other side's average visibility must lead before switching
    MEMORY = 0.9      # share of the running average kept at each frame

    def __init__(self):
        self.side, self.lead = None, 0.0     # lead > 0: left more visible

    def update(self, landmarks):
        now = float(landmarks[_LEFT, 3].mean() - landmarks[_RIGHT, 3].mean())
        if self.side is None:
            self.side, self.lead = ("left" if now >= 0 else "right"), now
        else:
            self.lead = self.MEMORY * self.lead + (1 - self.MEMORY) * now
            if self.side == "left" and self.lead < -self.MARGIN:
                self.side = "right"
            elif self.side == "right" and self.lead > self.MARGIN:
                self.side = "left"
        return self.side


# ------------------------------------------------------- camera position ---
# Measured on the test clips: filmed from the front, the turn is 1-10 degrees and the width
# 0.44-0.56; from the side, 80-88 degrees and 0.03-0.20. The limits below sit well inside the
# gap, and both measures have to agree, so a video is only called front or side when it is clear.
FRONT_TURN, SIDE_TURN = 30, 60        # degrees
FRONT_WIDTH, SIDE_WIDTH = 0.35, 0.25

_LS, _RS, _LH, _RH = (INDEX[n] for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))


def facing(image, world, width, height):
    """How one frame's person is turned relative to the camera, as (turn, width).

    turn: the angle between the line through the shoulders (and hips) and the picture, in
    degrees: 0 when facing the camera or with their back to it, 90 when side-on. It comes from
    MediaPipe's 3D landmarks, so it holds whether the person stands, bends or lies down.
    width: how wide the shoulders and hips look in the picture, as a share of the torso's
    length: about 0.5 from the front, close to 0 from the side.
    image / world: rows of landmarks as MediaPipe gives them (pose_landmarks, pose_world_landmarks).
    """
    turn = 0.0
    for a, b in ((_LS, _RS), (_LH, _RH)):
        dx, dy, dz = (float(world[b][i] - world[a][i]) for i in range(3))
        turn += math.degrees(math.atan2(abs(dz), math.hypot(dx, dy))) / 2
    px = [(float(image[i][0]) * width, float(image[i][1]) * height) for i in (_LS, _RS, _LH, _RH)]
    torso = geometry.distance(geometry.midpoint(px[0], px[1]), geometry.midpoint(px[2], px[3]))
    across = (geometry.distance(px[0], px[1]) + geometry.distance(px[2], px[3])) / 2
    return turn, across / max(torso, 1e-6)


def view(frames):
    """'front' or 'side' for a video, from the (turn, width) of several of its frames.

    None when it is in between, mixed, or the two measures disagree: only a clear answer
    should ever be used to turn a video away.
    """
    if not frames:
        return None
    turn = float(np.median([f[0] for f in frames]))
    width = float(np.median([f[1] for f in frames]))
    if turn <= FRONT_TURN and width >= FRONT_WIDTH:
        return "front"
    if turn >= SIDE_TURN and width <= SIDE_WIDTH:
        return "side"
    return None


# ------------------------------------------------------------------ cache ---

def _cache_file(video, options):
    folder = os.environ.get("FITCHECK_POSE_CACHE")
    if not folder or not video or not os.path.isfile(video):
        return None
    import mediapipe
    digest = hashlib.sha1(json.dumps([options, mediapipe.__version__], sort_keys=True).encode())
    with open(video, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return os.path.join(folder, digest.hexdigest()[:24] + ".npz")


def _to_array(landmark_list, width):
    """A MediaPipe landmark list as rows of x, y, z, visibility(, presence). None stays None."""
    if landmark_list is None:
        return None
    rows = [[p.x, p.y, p.z, p.visibility, p.presence if p.HasField("presence") else np.nan][:width]
            for p in landmark_list.landmark]
    return np.array(rows, np.float32)


def _to_landmarks(rows, cls):
    out = cls()
    for row in rows:
        p = out.landmark.add(x=float(row[0]), y=float(row[1]), z=float(row[2]), visibility=float(row[3]))
        if len(row) > 4 and not np.isnan(row[4]):
            p.presence = float(row[4])
    return out


_open = []


def save_all():
    """Write the landmark cache of every tracker still open, where one is in use.

    webapp/runner.py calls this before it exits: it leaves without Python's normal shutdown,
    and most analysers never close their tracker.
    """
    for tracker in list(_open):
        tracker.save()


# ---------------------------------------------------------------- tracker ---

class Tracker:
    """Drop-in for `mediapipe.solutions.pose.Pose`; see the module notes above.

    video: path of the video being analysed. Only needed for the landmark cache.
    Other arguments are passed to MediaPipe unchanged.
    """

    def __init__(self, video=None, **options):
        self.options = {**DEFAULTS, **options}
        self.body = None                 # the latest frame's pose, or None if nobody was found
        self._sides = SideChooser()
        self._model = None               # loaded on first use: a cached video never needs it
        self._file = _cache_file(video, self.options)
        self._saved = None               # landmarks from an earlier run of this same video
        self._fresh = []                 # landmarks from this run, to save when it ends
        self._frame = 0
        if self._file and os.path.exists(self._file):
            with np.load(self._file) as z:
                self._saved = (z["image"], z["world"])
        _open.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def process(self, rgb):
        """Find the pose in one RGB frame. Frames must be given in order, as MediaPipe needs."""
        height, width = rgb.shape[:2]
        if self._saved is not None:
            result, image = self._replay()
        else:
            if self._model is None:
                import mediapipe
                self._model = mediapipe.solutions.pose.Pose(**self.options)
            result = self._model.process(rgb)
            image = _to_array(result.pose_landmarks, 5)
            if self._file:
                self._fresh.append((image, _to_array(result.pose_world_landmarks, 4)))
        self._frame += 1
        self.body = None if image is None else Body(image, width, height, self._sides.update(image))
        return result

    def _replay(self):
        image, world = self._saved
        if self._frame >= len(image):
            raise RuntimeError(f"The saved pose tracking in {self._file} is shorter than the video. "
                               "Delete that file (or the whole cache folder) and run again.")
        from mediapipe.framework.formats import landmark_pb2
        rows = image[self._frame]
        if np.isnan(rows[0, 0]):
            return types.SimpleNamespace(pose_landmarks=None, pose_world_landmarks=None), None
        return types.SimpleNamespace(
            pose_landmarks=_to_landmarks(rows, landmark_pb2.NormalizedLandmarkList),
            pose_world_landmarks=_to_landmarks(world[self._frame], landmark_pb2.LandmarkList)), rows

    def save(self):
        """Write this run's landmarks to the cache, if a cache is in use."""
        if self._fresh and self._file:
            def stack(i, width):
                return np.stack([f[i] if f[i] is not None else np.full((len(NAMES), width), np.nan, np.float32)
                                 for f in self._fresh])
            os.makedirs(os.path.dirname(self._file), exist_ok=True)
            tmp = self._file[:-4] + ".tmp.npz"
            np.savez_compressed(tmp, image=stack(0, 5), world=stack(1, 4))
            os.replace(tmp, self._file)
            self._fresh = []

    def close(self):
        self.save()
        if self in _open:
            _open.remove(self)
        if self._model is not None:
            self._model.close()
            self._model = None
