"""Run an analyser script from analysers/ headlessly.

Usage: python runner.py <job_dir> <script.py> [script args...]

- cv2.imshow is replaced so no desktop window opens; instead the latest frame is
  saved to <job_dir>/preview.jpg and progress is printed as "@@PROGRESS <n>".
- cv2.waitKey never returns 'q', so the script runs to the end of the video.
- cv2.putText draws with a real font (see overlay_text.py) instead of Hershey.
"""
import os
import runpy
import sys
import time

import cv2

import overlay_text

job_dir, script, *script_args = sys.argv[1:]
_count = 0


def _imshow(_name, frame):
    global _count
    _count += 1
    if _count % 10 == 1:
        tmp = os.path.join(job_dir, "preview_tmp.jpg")
        cv2.imwrite(tmp, frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        os.replace(tmp, os.path.join(job_dir, "preview.jpg"))
        print(f"@@PROGRESS {_count}", flush=True)


overlay_text.install()
cv2.imshow = _imshow


# The analysers time things with time.time(), which makes results depend on how fast
# the server processes frames. Replace it with a clock that follows the video instead:
# each frame read advances it by 1/fps.
class _VideoClock:
    def __init__(self):
        self.start, self.now = _real_time(), 0.0

    def __call__(self):
        return self.start + self.now


_real_time = time.time
_clock = _VideoClock()
time.time = _clock


class _TimedCapture(cv2.VideoCapture):
    def read(self, *a, **k):
        ok, frame = super().read(*a, **k)
        if ok:
            fps = self.get(cv2.CAP_PROP_FPS) or 30
            _clock.now += 1.0 / fps
        return ok, frame


cv2.VideoCapture = _TimedCapture
cv2.waitKey = lambda *a, **k: -1
cv2.destroyAllWindows = lambda *a, **k: None
cv2.namedWindow = lambda *a, **k: None

sys.argv = [script, *script_args]
runpy.run_path(script, run_name="__main__")

# Exit straight away: MediaPipe sometimes segfaults while Python tears down at exit,
# after the analysis has already finished and saved its output.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
