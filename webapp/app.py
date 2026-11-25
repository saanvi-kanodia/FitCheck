"""FitCheck web app: upload a video, run a fitness test analysis, view results.

Run from the project root:
    .venv/bin/python webapp/app.py
then open http://127.0.0.1:5050
"""
import glob
import json
import shutil
import time
import os
import re
import subprocess
import sys
import threading
import uuid

import cv2
import imageio_ffmpeg
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from pdf_report import FILMING, _rating, _summarise, build_pdf

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
JOBS_DIR = os.path.join(APP_DIR, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

TESTS = {
    "pushup": {"name": "Push-ups", "script": "Pushup_analysis.py", "extra": ["-wt", "PushUp"]},
    "squats": {"name": "Squats", "script": "Squats_analysis.py"},
    "curlups": {"name": "Curl-ups", "script": "CurlUps_analysis.py"},
    "plank": {"name": "Plank hold", "script": "plankHold_analysis.py"},
    "jumpingjack": {"name": "Jumping jacks", "script": "jumpingJack_analysis.py"},
    "verticaljump": {"name": "Vertical jump", "script": "verticalJump_analysis.py"},
    "shuttlerun": {"name": "Shuttle run", "script": "shuttleRun_analysis.py"},
    "sitreach": {"name": "Sit and reach", "script": "SitReach_analysis.py", "needs_height": True},
    "backstretch": {"name": "Back stretch", "script": "backStretch_analysis.py"},
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
jobs = {}
SAVED_KEYS = ("status", "test", "test_key", "created", "report", "log", "saved", "video_name")


def write_meta(job_id):
    job = jobs[job_id]
    with open(os.path.join(JOBS_DIR, job_id, "job.json"), "w", encoding="utf-8") as f:
        json.dump({k: job.get(k) for k in SAVED_KEYS}, f)


def load_saved():
    """Restore saved analyses from disk so they survive a server restart."""
    for meta in glob.glob(os.path.join(JOBS_DIR, "*", "job.json")):
        try:
            with open(meta, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        if data.get("saved"):
            jobs[os.path.basename(os.path.dirname(meta))] = data


load_saved()


def run_job(job_id, test_key, video_path, height):
    job = jobs[job_id]
    job_dir = os.path.join(JOBS_DIR, job_id)
    test = TESTS[test_key]
    raw_out = os.path.join(job_dir, "analysed_raw.mp4")

    cap = cv2.VideoCapture(video_path)
    job["total"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()

    cmd = [sys.executable, "-u", os.path.join(APP_DIR, "runner.py"), job_dir,
           os.path.join(PROJECT_DIR, test["script"]), "-v", video_path, "-o", raw_out,
           *test.get("extra", [])]
    if test.get("needs_height"):
        cmd += ["--height", str(height)]

    # Reports are written to ~/Downloads; point HOME at the job dir to capture them.
    env = dict(os.environ, HOME=job_dir)
    os.makedirs(os.path.join(job_dir, "Downloads"), exist_ok=True)

    log = []
    proc = subprocess.Popen(cmd, cwd=PROJECT_DIR, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        m = re.match(r"@@PROGRESS (\d+)", line)
        if m:
            job["frame"] = int(m.group(1))
        elif not re.match(r"^[WI]\d{4} |WARNING: All log|INFO: Created", line):
            log.append(line.rstrip())
    proc.wait()
    job["log"] = "\n".join(l for l in log if l.strip())

    if proc.returncode != 0 or not os.path.exists(raw_out):
        job["status"] = "error"
        return

    # Re-encode to H.264 so browsers can play it.
    job["status"] = "encoding"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", raw_out,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    os.path.join(job_dir, "analysed.mp4")], check=False)

    reports = glob.glob(os.path.join(job_dir, "Downloads", "*.txt"))
    if reports:
        with open(reports[0], encoding="utf-8", errors="replace") as f:
            job["report"] = f.read()
    job["status"] = "done"
    write_meta(job_id)


@app.get("/")
def index():
    return render_template("index.html", tests=TESTS)


@app.post("/api/analyze")
def analyze():
    test_key = request.form.get("test")
    file = request.files.get("video")
    if test_key not in TESTS or not file or not file.filename:
        return jsonify(error="Choose a test and a video file."), 400
    height = request.form.get("height", type=float)
    if TESTS[test_key].get("needs_height") and not height:
        return jsonify(error="Sit and reach needs your height in cm."), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir)
    video_path = os.path.join(job_dir, secure_filename(file.filename) or "input.mp4")
    file.save(video_path)

    jobs[job_id] = {"status": "running", "frame": 0, "total": 0, "test": TESTS[test_key]["name"],
                    "test_key": test_key, "created": time.time(), "saved": False,
                    "video_name": file.filename}
    threading.Thread(target=run_job, args=(job_id, test_key, video_path, height), daemon=True).start()
    return jsonify(job=job_id)


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id) or abort(404)
    if job.get("status") != "done":
        return jsonify(job)
    summary = _summarise(job)
    summary["rating"] = _rating(summary["form"])
    summary["filming"] = FILMING.get(job.get("test_key") or "")
    return jsonify({**job, "summary": summary})


@app.post("/api/jobs/<job_id>/save")
def save_job(job_id):
    job = jobs.get(job_id) or abort(404)
    if job["status"] != "done":
        return jsonify(error="Only finished analyses can be saved."), 400
    job["saved"] = True
    write_meta(job_id)
    return jsonify(ok=True)


@app.delete("/api/jobs/<job_id>")
def delete_job(job_id):
    if job_id not in jobs or not re.fullmatch(r"[0-9a-f]{12}", job_id):
        abort(404)
    if jobs[job_id]["status"] in ("running", "encoding"):
        return jsonify(error="Wait for the analysis to finish."), 400
    jobs.pop(job_id)
    shutil.rmtree(os.path.join(JOBS_DIR, job_id), ignore_errors=True)
    return jsonify(ok=True)


@app.get("/api/history")
def history():
    items = [{"id": k, "test": j.get("test"), "test_key": j.get("test_key"), "created": j.get("created"),
              "video_name": j.get("video_name")}
             for k, j in jobs.items() if j.get("saved")]
    items.sort(key=lambda x: x["created"] or 0, reverse=True)
    return jsonify(items)


@app.get("/jobs/<job_id>/report.pdf")
def job_report(job_id):
    job = jobs.get(job_id) or abort(404)
    if job.get("status") not in ("done", "error"):
        abort(409)
    job_dir = os.path.join(JOBS_DIR, job_id)
    pdf_path = os.path.join(job_dir, "report.pdf")
    if not os.path.exists(pdf_path):
        build_pdf(job, job_dir, pdf_path)
    name = f"{job.get('test_key', 'analysis')}_report_{time.strftime('%Y%m%d_%H%M', time.localtime(job.get('created') or 0))}.pdf"
    as_download = request.args.get("download") == "1"
    resp = send_from_directory(job_dir, "report.pdf", mimetype="application/pdf",
                               as_attachment=as_download, download_name=name)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/jobs/<job_id>/<name>")
def job_file(job_id, name):
    if job_id not in jobs or name not in {"preview.jpg", "analysed.mp4"}:
        abort(404)
    resp = send_from_directory(os.path.join(JOBS_DIR, job_id), name)
    resp.headers["Cache-Control"] = "no-store"
    return resp


if __name__ == "__main__":
    app.run(debug=False, port=5050, threaded=True)
