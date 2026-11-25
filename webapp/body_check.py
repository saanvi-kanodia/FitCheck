"""Check that a whole body (head to ankles) is visible in a video before analysing it.

Usage: python body_check.py <video>
Prints one JSON line: {"ok": bool, "message": str}
"""
import json
import sys

import cv2
import mediapipe as mp

SAMPLES = 24          # frames checked, spread across the video
MIN_VIS = 0.5         # landmark visibility needed to count as seen
MIN_SECONDS = 2       # shorter clips can't show a full rep
MIN_GOOD = 0.6        # share of frames with a person that must show the whole body
P = mp.solutions.pose.PoseLandmark
# One of each pair is enough: in side-on footage the far limb is often hidden.
GROUPS = {
    "head": [P.NOSE, P.LEFT_EAR, P.RIGHT_EAR],
    "shoulders": [P.LEFT_SHOULDER, P.RIGHT_SHOULDER],
    "hips": [P.LEFT_HIP, P.RIGHT_HIP],
    "knees": [P.LEFT_KNEE, P.RIGHT_KNEE],
    "ankles": [P.LEFT_ANKLE, P.RIGHT_ANKLE],
}


def seen(lm, idx):
    p = lm[idx.value]
    return p.visibility >= MIN_VIS and -0.02 <= p.x <= 1.02 and -0.02 <= p.y <= 1.02


def main(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    readable, _ = cap.read()
    if not cap.isOpened() or not readable or total <= 0:
        return {"ok": False, "message": "This file couldn't be read as a video. Please upload an MP4 or MOV file."}
    if fps and total / fps < MIN_SECONDS:
        return {"ok": False, "message": f"This video is only {total / fps:.1f} seconds long. "
                f"Please upload at least {MIN_SECONDS} seconds showing the full movement."}
    frames = sorted({int(i * (total - 1) / max(1, SAMPLES - 1)) for i in range(SAMPLES)}) if total else []
    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
    people, good, missing = 0, 0, {}
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            continue
        people += 1
        lm = res.pose_landmarks.landmark
        gone = [g for g, ids in GROUPS.items() if not any(seen(lm, i) for i in ids)]
        if gone:
            for g in gone:
                missing[g] = missing.get(g, 0) + 1
        else:
            good += 1

    if people < max(3, len(frames) * 0.3):
        return {"ok": False, "message": "We couldn't find a person in this video. Make sure you're clearly "
                "visible, with good lighting, and upload it again."}
    if good / people < MIN_GOOD:
        worst = sorted(missing, key=missing.get, reverse=True)[:2]
        return {"ok": False, "message": "Your whole body needs to be in the video, from head to feet. "
                f"Your {' and '.join(worst)} were out of frame in much of this clip. "
                "Step back or move the camera, then upload a new video."}
    return {"ok": True, "message": ""}


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1])))
