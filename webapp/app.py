"""FitCheck web app: upload a video, run a fitness test analysis, view results.

Run from the project root:
    .venv/bin/python webapp/app.py
then open http://127.0.0.1:5050

Each browser gets a private id (cookie). Jobs belong to that id and only it can see them.
Finished results are copied into the browser's own storage, which is the user's history;
the server keeps jobs only for a limited time.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import cv2
import imageio_ffmpeg
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from pdf_report import FILMING, _rating, _summarise, build_pdf
import leaderboard

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
ANALYSERS_DIR = os.path.join(PROJECT_DIR, "analysers")
# Where jobs/ and uploads/ are kept. tests/eval points this at a temporary folder so that
# measuring accuracy never touches real jobs.
DATA_DIR = os.environ.get("FITCHECK_DATA_DIR") or APP_DIR
JOBS_DIR = os.path.join(DATA_DIR, "jobs")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
# MediaPipe imports matplotlib, which lists every font on the machine the first time it is used
# and keeps that list in its settings folder. The analysers run with HOME pointed at the job
# folder, so the list used to be rebuilt for every single analysis (15 seconds on a Mac).
# One fixed folder means it is built once.
MPL_DIR = os.environ.get("MPLCONFIGDIR") or os.path.join(tempfile.gettempdir(), "fitcheck-matplotlib")
for d in (JOBS_DIR, UPLOADS_DIR, MPL_DIR):
    os.makedirs(d, exist_ok=True)

TESTS = {
    "pushup": {"name": "Push-ups", "script": "pushup.py", "extra": ["-wt", "PushUp"]},
    "squats": {"name": "Squats", "script": "squats.py"},
    "curlups": {"name": "Curl-ups", "script": "curl_ups.py"},
    "plank": {"name": "Plank hold", "script": "plank_hold.py"},
    "jumpingjack": {"name": "Jumping jacks", "script": "jumping_jack.py"},
    "verticaljump": {"name": "Vertical jump", "script": "vertical_jump.py"},
    "shuttlerun": {"name": "Shuttle run", "script": "shuttle_run.py"},
    "sitreach": {"name": "Sit and reach", "script": "sit_reach.py", "needs_height": True},
    "backstretch": {"name": "Back stretch", "script": "back_stretch.py"},
}

MAX_MB = 50                      # decimal megabytes, as phones and computers show file sizes
MAX_BYTES = MAX_MB * 1000 * 1000
CHUNK_BYTES = 2 * 1000 * 1000    # upload piece size (client sends this much per request)
MAX_SIDE = 640                   # longest side of the video passed to the analysers
KEEP_HOURS = 24                  # server copies are removed after this; history lives in the browser
ID = re.compile(r"[0-9a-f]{12}")
ACTIVE = ("queued", "running", "encoding")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = CHUNK_BYTES + 64 * 1024
jobs = {}
uploads = {}
lock = threading.Lock()
META_KEYS = ("status", "test", "test_key", "created", "report", "log", "video_name", "owner",
             "message", "height", "size", "seconds")


def _memory_mb():
    """Memory available to this container, if it is limited (cgroup v2 / v1)."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(path).read().strip()
            if v.isdigit() and int(v) < 1 << 50:
                return int(v) // (1024 * 1024)
        except OSError:
            pass
    return None


def _parallel():
    """How many analyses can run at once. One analysis peaks around 300 MB."""
    if os.environ.get("FITCHECK_PARALLEL"):
        return max(1, int(os.environ["FITCHECK_PARALLEL"]))
    mem = _memory_mb()
    by_mem = max(1, (mem - 150) // 320) if mem else 4
    return max(1, min(by_mem, os.cpu_count() or 1, 4))


PARALLEL = _parallel()
RUN_SLOT = threading.Semaphore(PARALLEL)
queue = []   # job ids waiting for a slot, in order


# ---------------------------------------------------------------- owners ----

def owner():
    return request.cookies.get("fc_owner") if ID.fullmatch(request.cookies.get("fc_owner") or "") else None


@app.after_request
def set_owner(resp):
    if not owner():
        resp.set_cookie("fc_owner", uuid.uuid4().hex[:12], max_age=3600 * 24 * 365 * 5,
                        httponly=True, samesite="Lax", secure=request.is_secure)
    return resp


def own_job(job_id):
    job = jobs.get(job_id) if ID.fullmatch(job_id or "") else None
    if not job or not owner() or job.get("owner") != owner():
        abort(404)
    return job


# ------------------------------------------------------------- persistence --

def write_meta(job_id):
    job = jobs[job_id]
    path = os.path.join(JOBS_DIR, job_id, "job.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump({k: job.get(k) for k in META_KEYS}, f)
    os.replace(path + ".tmp", path)


def load_jobs():
    """Restore jobs after a restart. Ones that were mid-analysis are re-queued from their upload."""
    for meta in glob.glob(os.path.join(JOBS_DIR, "*", "job.json")):
        job_id = os.path.basename(os.path.dirname(meta))
        try:
            with open(meta, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        jobs[job_id] = data
        if data.get("status") in ACTIVE:
            src = _source(job_id)
            if src:
                start_job(job_id, src)
            else:
                data.update(status="error", message="The server restarted during this analysis.")


def _source(job_id):
    for p in glob.glob(os.path.join(JOBS_DIR, job_id, "source.*")):
        return p
    return None


def cleanup():
    """Remove old server copies (the browser keeps the user's history) and stale uploads."""
    cutoff = time.time() - KEEP_HOURS * 3600
    for job_id, job in list(jobs.items()):
        if job.get("status") not in ACTIVE and (job.get("created") or 0) < cutoff:
            jobs.pop(job_id, None)
            shutil.rmtree(os.path.join(JOBS_DIR, job_id), ignore_errors=True)
    for up_id in os.listdir(UPLOADS_DIR):
        path = os.path.join(UPLOADS_DIR, up_id)
        if os.path.getmtime(path) < cutoff:
            uploads.pop(up_id, None)
            shutil.rmtree(path, ignore_errors=True)


def _cleanup_loop():
    while True:
        time.sleep(3600)
        try:
            cleanup()
        except Exception:
            pass


# ---------------------------------------------------------------- running ---

def start_job(job_id, src):
    job = jobs[job_id]
    job.update(status="queued", frame=0, total=0, stage=None, message=None)
    with lock:
        queue.append(job_id)
    write_meta(job_id)
    threading.Thread(target=run_job, args=(job_id, src), daemon=True).start()


def run_job(job_id, src):
    job = jobs[job_id]
    with RUN_SLOT:
        with lock:
            if job_id in queue:
                queue.remove(job_id)
        job["status"] = "running"
        write_meta(job_id)
        try:
            _run_job(job_id, src)
        except Exception as e:  # never leave a job stuck in "running"
            job["status"] = "error"
            job["log"] = (job.get("log") or "") + f"\nInternal error: {e}"
        write_meta(job_id)


def _run_job(job_id, video_path):
    job = jobs[job_id]
    job_dir = os.path.join(JOBS_DIR, job_id)
    test = TESTS[job["test_key"]]
    raw_out = os.path.join(job_dir, "analysed_raw.mp4")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    child_env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MPLCONFIGDIR=MPL_DIR)

    # Shrink large videos first: pose tracking doesn't need more than this, and it keeps
    # memory within small servers (Render's free plan has 512 MB).
    job["stage"] = "preparing"
    small = os.path.join(job_dir, "input_small.mp4")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", video_path,
                    "-vf", f"scale='if(gt(iw,ih),min({MAX_SIDE},iw),-2)':'if(gt(iw,ih),-2,min({MAX_SIDE},ih))'",
                    "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-threads", "1", small], check=False)
    if os.path.exists(small) and os.path.getsize(small) > 0:
        video_path = small

    # Refuse videos that don't show the whole body, or are filmed from the wrong side for the
    # test: the analysers give nonsense otherwise.
    job["stage"] = "checking"
    check = subprocess.run([sys.executable, os.path.join(APP_DIR, "body_check.py"), video_path, job["test_key"]],
                           capture_output=True, text=True, env=child_env)
    try:
        verdict = json.loads(check.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        verdict = {"ok": True}
    job["stage"] = None
    if not verdict.get("ok"):
        job["status"] = "rejected"
        job["message"] = verdict.get("message")
        return

    cap = cv2.VideoCapture(video_path)
    job["total"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    job["seconds"] = round(job["total"] / fps, 1) if fps else None   # timed challenges check this
    cap.release()

    cmd = [sys.executable, "-u", os.path.join(APP_DIR, "runner.py"), job_dir,
           os.path.join(ANALYSERS_DIR, test["script"]), "-v", video_path, "-o", raw_out,
           *test.get("extra", [])]
    if test.get("needs_height"):
        cmd += ["--height", str(job.get("height"))]

    # Reports are written to ~/Downloads; point HOME at the job dir to capture them.
    for old in glob.glob(os.path.join(job_dir, "Downloads", "*")):
        os.remove(old)
    os.makedirs(os.path.join(job_dir, "Downloads"), exist_ok=True)
    log = []
    proc = subprocess.Popen(cmd, cwd=PROJECT_DIR, env=dict(child_env, HOME=job_dir), stdout=subprocess.PIPE,
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
        job["message"] = "The analysis stopped unexpectedly."
        return

    # Re-encode to H.264 so browsers can play it.
    job["status"] = "encoding"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", raw_out,
                    "-c:v", "libx264", "-preset", "veryfast", "-threads", "1", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", os.path.join(job_dir, "analysed.mp4")], check=False)
    reports = glob.glob(os.path.join(job_dir, "Downloads", "*.txt"))
    if reports:
        with open(reports[0], encoding="utf-8", errors="replace") as f:
            job["report"] = f.read()
    # Build the PDF now so it is ready for the browser to keep.
    try:
        build_pdf(job, job_dir, os.path.join(job_dir, "report.pdf"))
    except Exception as e:
        job["log"] += f"\nPDF error: {e}"
    for tmp in ("analysed_raw.mp4", "input_small.mp4"):
        try:
            os.remove(os.path.join(job_dir, tmp))
        except OSError:
            pass
    job["status"] = "done"


def public(job_id, job):
    """What the browser sees for a job."""
    out = {k: job.get(k) for k in ("status", "test", "test_key", "created", "video_name", "frame", "total",
                                   "stage", "message", "size")}
    out["id"] = job_id
    if job.get("status") == "queued":
        # Analyses that must finish (or start) before this one: those queued earlier, plus
        # any running ones while every slot is busy.
        with lock:
            before = queue.index(job_id) if job_id in queue else 0
        running = sum(1 for j in jobs.values() if j.get("status") in ("running", "encoding"))
        out["ahead"] = before + (running if running >= PARALLEL else 0)
    if job.get("status") == "done":
        summary = _summarise(job)
        summary["rating"] = _rating(summary["form"])
        summary["filming"] = FILMING.get(job.get("test_key") or "")
        out["summary"] = summary
        out["report"] = job.get("report")
        out["has_pdf"] = os.path.exists(os.path.join(JOBS_DIR, job_id, "report.pdf"))
        # A signed copy of the result, so the browser can later enter it in a challenge without being
        # able to change the number.
        m = re.search(r"-?\d+(\.\d+)?", (summary.get("main") or ("", ""))[1] or "")
        if m:
            out["score_token"] = leaderboard.sign(DATA_DIR, {"test": job.get("test_key"), "value": float(m.group()),
                                                            "seconds": job.get("seconds"), "job": job_id,
                                                            "week": leaderboard.week_key()})
    if job.get("status") == "error":
        out["log"] = job.get("log")
    return out


# ------------------------------------------------------------------ pages ---

@app.get("/")
def index():
    samples = [k for k in TESTS if os.path.exists(os.path.join(APP_DIR, "static", f"sample-{k}.mp4"))]
    return render_template("index.html", tests=TESTS, filming=FILMING, max_mb=MAX_MB, samples=samples,
                           chunk=CHUNK_BYTES)


@app.errorhandler(413)
def too_large(_e):
    return jsonify(error=f"Video is over {MAX_MB} MB. Trim it to under a minute and try again."), 413


# ---------------------------------------------------------------- uploads ---
# A video is uploaded in pieces so an interrupted upload can continue where it stopped.

def _upload(up_id):
    up = uploads.get(up_id) if ID.fullmatch(up_id or "") else None
    if not up:
        meta = os.path.join(UPLOADS_DIR, up_id or "x", "meta.json")
        if ID.fullmatch(up_id or "") and os.path.exists(meta):
            with open(meta, encoding="utf-8") as f:
                up = uploads[up_id] = json.load(f)
    if not up or up.get("owner") != owner():
        abort(404)
    return up


def _received(up_id):
    try:
        return os.path.getsize(os.path.join(UPLOADS_DIR, up_id, "data"))
    except OSError:
        return 0


@app.post("/api/uploads")
def create_upload():
    data = request.get_json(silent=True) or {}
    size, test_key = data.get("size"), data.get("test")
    if test_key not in TESTS or not isinstance(size, int) or size <= 0 or not data.get("name"):
        return jsonify(error="Choose a test and a video file."), 400
    if size > MAX_BYTES:
        return jsonify(error=f"Video is over {MAX_MB} MB. Trim it to under a minute and try again."), 413
    height = data.get("height")
    if TESTS[test_key].get("needs_height"):
        try:
            height = float(height)
            assert 80 <= height <= 250
        except (TypeError, ValueError, AssertionError):
            return jsonify(error="Sit and reach needs your height in cm (80 to 250)."), 400
    if not owner():
        return jsonify(error="Please reload the page and try again."), 400
    up_id = uuid.uuid4().hex[:12]
    os.makedirs(os.path.join(UPLOADS_DIR, up_id))
    open(os.path.join(UPLOADS_DIR, up_id, "data"), "wb").close()
    up = uploads[up_id] = {"owner": owner(), "size": size, "name": str(data["name"])[:200],
                           "test": test_key, "height": height}
    with open(os.path.join(UPLOADS_DIR, up_id, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(up, f)
    return jsonify(upload=up_id, received=0, chunk=CHUNK_BYTES)


@app.get("/api/uploads/<up_id>")
def upload_status(up_id):
    up = _upload(up_id)
    return jsonify(upload=up_id, received=_received(up_id), size=up["size"], job=up.get("job"))


@app.put("/api/uploads/<up_id>")
def upload_chunk(up_id):
    up = _upload(up_id)
    if up.get("job"):
        return jsonify(received=up["size"], job=up["job"])
    offset = request.args.get("offset", type=int)
    have = _received(up_id)
    body = request.get_data()
    if offset is None or offset != have:      # client is out of step: tell it where to continue
        return jsonify(received=have, error="offset mismatch"), 409
    if have + len(body) > up["size"]:
        return jsonify(error="More data than the file size."), 400
    with open(os.path.join(UPLOADS_DIR, up_id, "data"), "ab") as f:
        f.write(body)
    return jsonify(received=have + len(body))


@app.post("/api/uploads/<up_id>/complete")
def complete_upload(up_id):
    up = _upload(up_id)
    if up.get("job"):
        return jsonify(job=up["job"])
    if _received(up_id) != up["size"]:
        return jsonify(error="The upload is incomplete.", received=_received(up_id)), 409
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir)
    ext = os.path.splitext(secure_filename(up["name"]))[1].lower() or ".mp4"
    src = os.path.join(job_dir, "source" + (ext if re.fullmatch(r"\.[a-z0-9]{1,5}", ext) else ".mp4"))
    shutil.move(os.path.join(UPLOADS_DIR, up_id, "data"), src)
    jobs[job_id] = {"test": TESTS[up["test"]]["name"], "test_key": up["test"], "created": time.time(),
                    "video_name": up["name"], "owner": owner(), "height": up.get("height"), "size": up["size"]}
    up["job"] = job_id
    with open(os.path.join(UPLOADS_DIR, up_id, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(up, f)
    start_job(job_id, src)
    return jsonify(job=job_id)


# ------------------------------------------------------------------- jobs ---

@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    return jsonify(public(job_id, own_job(job_id)))


@app.post("/api/jobs/<job_id>/retry")
def retry_job(job_id):
    job = own_job(job_id)
    if job.get("status") in ACTIVE:
        return jsonify(job=job_id)
    src = _source(job_id)
    if not src:
        return jsonify(error="The original video is no longer on the server."), 410
    start_job(job_id, src)
    return jsonify(job=job_id)


@app.delete("/api/jobs/<job_id>")
def delete_job(job_id):
    job = own_job(job_id)
    if job.get("status") in ACTIVE:
        return jsonify(error="Wait for the analysis to finish."), 400
    jobs.pop(job_id, None)
    shutil.rmtree(os.path.join(JOBS_DIR, job_id), ignore_errors=True)
    return jsonify(ok=True)


@app.get("/api/active")
def active_jobs():
    """This browser's jobs still on the server, newest first (used to restore after a refresh)."""
    me = owner()
    items = [public(k, j) for k, j in jobs.items() if me and j.get("owner") == me]
    items.sort(key=lambda x: x.get("created") or 0, reverse=True)
    return jsonify(items)


@app.get("/jobs/<job_id>/report.pdf")
def job_report(job_id):
    job = own_job(job_id)
    if job.get("status") != "done":
        abort(409)
    job_dir = os.path.join(JOBS_DIR, job_id)
    pdf_path = os.path.join(job_dir, "report.pdf")
    if not os.path.exists(pdf_path):
        build_pdf(job, job_dir, pdf_path)
    name = f"{job.get('test_key', 'analysis')}_report_{time.strftime('%Y%m%d_%H%M', time.localtime(job.get('created') or 0))}.pdf"
    resp = send_from_directory(job_dir, "report.pdf", mimetype="application/pdf",
                               as_attachment=request.args.get("download") == "1", download_name=name)
    resp.headers["Cache-Control"] = "private, no-store"
    return resp


@app.get("/jobs/<job_id>/<name>")
def job_file(job_id, name):
    own_job(job_id)
    if name not in {"preview.jpg", "analysed.mp4"}:
        abort(404)
    resp = send_from_directory(os.path.join(JOBS_DIR, job_id), name)
    resp.headers["Cache-Control"] = "private, no-store"
    return resp


# ------------------------------------------------------------ leaderboard --

_store = None


def store():
    global _store
    if _store is None:
        _store = leaderboard.Store(DATA_DIR)
        # FITCHECK_SAMPLE_SCORES=1 fills the boards with labelled sample scores (refreshed on each
        # start, so the weekly board always has some). Real scores are never touched.
        # Anything else removes them again (only rows marked as samples; real scores are never touched).
        from seed_leaderboard import seed
        seed(_store, clear_only=os.environ.get("FITCHECK_SAMPLE_SCORES") != "1")
    return _store


@app.get("/api/leaderboard")
def get_leaderboard():
    return jsonify(challenges=leaderboard.board(store(), request.args.get("name")))


@app.post("/api/leaderboard")
def post_leaderboard():
    data = request.get_json(silent=True) or {}
    ok, message, rank = leaderboard.submit(store(), DATA_DIR, data.get("token"), data.get("challenge"), data.get("name"),
                                           live=data.get("live") if isinstance(data.get("live"), dict) else None)
    return (jsonify(ok=True, message=message, rank=rank) if ok else (jsonify(error=message), 400))


@app.post("/api/live-report")
def live_report():
    """PDF for a live camera session. The browser sends its summary; nothing is stored."""
    data = request.get_json(silent=True) or {}
    key = data.get("test_key")
    if key not in TESTS:
        abort(400)

    def pairs(v, n=12):
        return [[str(a)[:80], str(b)[:80]] for a, b in (v or [])[:n]
                if isinstance(v, list)] if isinstance(v, list) else []
    try:
        main = data.get("main")
        summary = {
            "main": [str(main[0])[:60], str(main[1])[:30]] if isinstance(main, list) and len(main) == 2 else None,
            "highlights": pairs(data.get("highlights")),
            "issues": pairs(data.get("issues")),
            "tips": [str(t)[:200] for t in (data.get("tips") or [])[:8]],
            "duration": float(data["duration"]) if data.get("duration") is not None else None,
            "form": max(0.0, min(4.0, float(data["form"]))) if data.get("form") is not None else None,
        }
    except (TypeError, ValueError, IndexError):
        abort(400)
    job = {"test": TESTS[key]["name"] + " (live)", "test_key": key, "report": "live",
           "created": float(data.get("created") or time.time())}
    tmp = os.path.join(UPLOADS_DIR, "live-" + uuid.uuid4().hex[:12])
    os.makedirs(tmp)
    try:
        path = os.path.join(tmp, "report.pdf")
        build_pdf(job, tmp, path, summary=summary)
        with open(path, "rb") as f:
            pdf = f.read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return app.response_class(pdf, mimetype="application/pdf")


load_jobs()
cleanup()
threading.Thread(target=_cleanup_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(debug=False, port=5050, threaded=True)
