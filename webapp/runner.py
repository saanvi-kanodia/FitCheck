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
cv2.waitKey = lambda *a, **k: -1
cv2.destroyAllWindows = lambda *a, **k: None
cv2.namedWindow = lambda *a, **k: None

sys.argv = [script, *script_args]
runpy.run_path(script, run_name="__main__")
