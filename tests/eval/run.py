"""Measure FitCheck's accuracy and speed on a set of labelled clips.

Every clip in manifest.json goes through the same steps as a real upload (shrink, body check,
analyser, encode, PDF), and what comes out is compared with the answer a person worked out
by watching the clip.

    .venv/bin/python tests/eval/run.py                      # run everything, print a table
    .venv/bin/python tests/eval/run.py --save baseline      # also keep the results
    .venv/bin/python tests/eval/run.py --against baseline   # show what changed since then
    .venv/bin/python tests/eval/run.py --test squats        # one test only
    .venv/bin/python tests/eval/run.py --clip plank-1       # clips whose path contains this

Exit code is 1 if an analysis crashed, or a clip whose answer has been confirmed came out wrong.
"""
import argparse
import json
import os
import re
import resource
import shutil
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(HERE, "results")
EPS = 1e-9
STAGES = ("preparing", "checking", "analysing", "encoding")


def video_seconds(path):
    import cv2
    cap = cv2.VideoCapture(path)
    frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    return frames / fps if frames else 0.0


def number(text):
    m = re.search(r"-?\d+(?:\.\d+)?", str(text or ""))
    return float(m.group()) if m else None


def fmt(v):
    return "-" if v is None else f"{v:g}"


def peak_child_mb():
    """Most memory any one subprocess has used so far (macOS reports bytes, Linux kilobytes)."""
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def run_clip(app, pdf_report, entry, job_id):
    """Run one clip through the upload pipeline and return what came out."""
    src = os.path.join(ROOT, entry["clip"])
    os.makedirs(os.path.join(app.JOBS_DIR, job_id))
    job = app.jobs[job_id] = {"test": app.TESTS[entry["test"]]["name"], "test_key": entry["test"],
                              "created": time.time(), "height": entry.get("height"), "owner": "eval",
                              "status": "running"}
    stages, stop = {}, threading.Event()

    def watch():   # time spent in each step, read from the progress fields the web page polls
        t = time.time()
        while not stop.wait(0.02):
            now = time.time()
            name = job.get("stage") or {"running": "analysing", "encoding": "encoding"}.get(job.get("status"))
            if name:
                stages[name] = stages.get(name, 0.0) + now - t
            t = now

    watcher = threading.Thread(target=watch, daemon=True)
    started = time.time()
    watcher.start()
    try:
        app._run_job(job_id, src)
    except Exception as e:
        job["status"], job["message"] = "error", f"{type(e).__name__}: {e}"
    took = time.time() - started
    stop.set()
    watcher.join()

    out = {"status": job["status"], "message": job.get("message"), "seconds": round(took, 1),
           "video_seconds": round(video_seconds(src), 1), "stages": {k: round(v, 1) for k, v in stages.items()},
           "value": None, "form": None, "figures": {}, "issues": None}
    if job["status"] == "done":
        s = pdf_report._summarise(job)
        out["value"] = number(s["main"][1]) if s["main"] else None
        out["form"] = s["form"]
        out["figures"] = {label.lower(): number(value) for label, value in s["highlights"] if number(value) is not None}
        out["issues"] = sorted({i[0] for i in s["issues"]})
    elif job["status"] == "error":
        out["log"] = (job.get("log") or "")[-800:]
    app.jobs.pop(job_id, None)
    shutil.rmtree(os.path.join(app.JOBS_DIR, job_id), ignore_errors=True)
    return out


def judge(entry, got):
    """Compare the main result with the clip's answer.

    Returns (result, error). result is 'ok', 'wrong', or None when the clip has no answer to
    compare with; error is how far the number was off, where there is one.
    """
    want = entry.get("expect_status", "done")
    if got["status"] == "error" and want != "error":
        return "wrong", None                      # a crash is wrong whatever the answer is
    if entry.get("expected") is None and "expect_status" not in entry:
        return None, None
    if got["status"] != want:
        return "wrong", None
    if want != "done":
        return "ok", None
    if got["value"] is None:
        return "wrong", None
    err = got["value"] - entry["expected"]
    return ("ok" if abs(err) <= entry.get("tolerance", 0) + EPS else "wrong"), err


def extras(entry, got):
    """Other labelled figures, such as reps attempted: [(label, found, wanted, correct)]."""
    out = []
    for label, want in (entry.get("expected_extra") or {}).items():
        found = got["figures"].get(label.lower())
        out.append((label, found, want, found is not None and abs(found - want) <= EPS))
    return out


def issue_score(entry, got):
    """Form problems as (flagged correctly, missed, false alarms); None if the clip has no labels for them."""
    if entry.get("expected_issues") is None or got["issues"] is None:
        return None
    want, found = {i.lower() for i in entry["expected_issues"]}, {i.lower() for i in got["issues"]}
    return len(want & found), len(want - found), len(found - want)


def changes(old, new):
    """What differs between an earlier result for a clip and this one, in words."""
    out = []
    if old.get("status") != new["status"]:
        out.append(f"{old.get('status')} -> {new['status']}")
    pairs = [("result", old.get("value"), new["value"]), ("form", old.get("form"), new["form"])]
    labels = sorted(set(old.get("figures") or {}) | set(new["figures"]))
    pairs += [(k, (old.get("figures") or {}).get(k), new["figures"].get(k)) for k in labels]
    for name, a, b in pairs:
        same = a == b or (a is not None and b is not None and abs(a - b) < 0.005)
        if not same:
            out.append(f"{name} {fmt(a)} -> {fmt(b)}")
    if (old.get("issues") or []) != (new["issues"] or []):
        out.append("form problems changed")
    return ", ".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test", help="only clips for this test key (e.g. squats)")
    ap.add_argument("--clip", help="only clips whose path contains this text")
    ap.add_argument("--save", metavar="NAME", help="keep results as results/NAME.json (full runs only)")
    ap.add_argument("--against", metavar="NAME", help="compare with results/NAME.json")
    args = ap.parse_args()

    with open(os.path.join(HERE, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)["clips"]
    clips, left_out = [], []
    for c in manifest:
        if (args.test and c["test"] != args.test) or (args.clip and args.clip not in c["clip"]):
            continue
        if c.get("skip") and not args.clip:
            left_out.append(f"{c['clip']}: {c['skip']}")
        elif not os.path.exists(os.path.join(ROOT, c["clip"])):
            left_out.append(f"{c['clip']}: FILE NOT FOUND")
        else:
            clips.append(c)
    if not clips:
        sys.exit("No clips to run.\n  " + "\n  ".join(left_out))
    partial = bool(args.test or args.clip or any("NOT FOUND" in x for x in left_out))
    if args.save and partial:
        sys.exit("--save keeps a full run only: drop --test/--clip and make sure every clip file exists.")

    before = None
    if args.against:
        with open(os.path.join(RESULTS, args.against + ".json"), encoding="utf-8") as f:
            before = json.load(f)

    # Borrow the web app's own pipeline, with its jobs kept in a throwaway folder so that
    # real jobs in webapp/jobs are never touched.
    work = tempfile.mkdtemp(prefix="fitcheck-eval-")
    os.environ["FITCHECK_DATA_DIR"] = work
    sys.path.insert(0, os.path.join(ROOT, "webapp"))
    import app
    import pdf_report

    results, failed = {}, 0
    try:
        for n, c in enumerate(clips):
            print(f"  [{n + 1}/{len(clips)}] {c['clip']} ...", file=sys.stderr, flush=True)
            got = run_clip(app, pdf_report, c, f"{n:012x}")
            result, err = judge(c, got)
            confirmed = bool(c.get("verified"))
            failed += result == "wrong" and (confirmed or got["status"] == "error")
            results[c["clip"]] = dict(got, test=c["test"], result=result, error=err, confirmed=confirmed,
                                      extras=extras(c, got), issue_score=issue_score(c, got))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # ---- table: one row per clip
    by_clip = {c["clip"]: c for c in clips}
    head = ["test", "clip", "expected", "got", "result", "other figures", "form problems", "answer", "time"]
    if before:
        head.append("change")
    rows = []
    for clip, r in results.items():
        c = by_clip[clip]
        if c.get("expect_status", "done") != "done":
            want = {"rejected": "refuse"}.get(c["expect_status"], c["expect_status"])
        else:
            want = fmt(c.get("expected")) + (f" ±{c['tolerance']:g}" if c.get("tolerance") else "")
        shown = fmt(r["value"]) if r["status"] == "done" else {"rejected": "refused"}.get(r["status"], r["status"])
        verdict = {"ok": "ok", "wrong": "WRONG", None: "-"}[r["result"]]
        if r["error"] is not None and r["result"] == "wrong":
            verdict += f" ({r['error']:+g})"
        other = "; ".join(f"{label}: {fmt(found)}" + ("" if ok else f" (should be {want_:g})")
                          for label, found, want_, ok in r["extras"])
        sc = r["issue_score"]
        form = "" if sc is None else "ok" if sc[1] == sc[2] == 0 else ", ".join(
            x for x in (f"{sc[1]} missed" if sc[1] else "", f"{sc[2]} false alarm{'s' * (sc[2] != 1)}" if sc[2] else "") if x)
        speed = r["video_seconds"] / r["seconds"] if r["seconds"] else 0
        row = [r["test"], os.path.basename(clip), want, shown, verdict, other, form,
               "confirmed" if r["confirmed"] else "proposed", f"{r['seconds']:.0f}s ({speed:.2f}x)"]
        if before:
            old = before["clips"].get(clip)
            row.append("new clip" if old is None else changes(old, r))
        rows.append(row)
    widths = [max(len(str(r[i])) for r in rows + [head]) for i in range(len(head))]
    for r in [head] + rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)).rstrip())

    # ---- per-test accuracy
    print()
    for test in sorted({r["test"] for r in results.values()}):
        rs = [r for r in results.values() if r["test"] == test]
        scored = [r for r in rs if r["result"]]
        if scored:
            line = f"{test}: {sum(r['result'] == 'ok' for r in scored)}/{len(scored)} clips correct"
            errs = [abs(r["error"]) for r in scored if r["error"] is not None]
            if errs:
                line += f", mean error {sum(errs) / len(errs):.2f}"
        else:
            line = f"{test}: no answers to compare with"
        ex = [e for r in rs for e in r["extras"]]
        if ex:
            line += f"; other figures {sum(e[3] for e in ex)}/{len(ex)} correct"
        sc = [r["issue_score"] for r in rs if r["issue_score"]]
        if sc:
            line += (f"; form problems: {sum(s[0] for s in sc)} flagged correctly, "
                     f"{sum(s[1] for s in sc)} missed, {sum(s[2] for s in sc)} false alarms")
        waiting = sum(1 for r in scored if not r["confirmed"])
        if waiting:
            line += f"  [{waiting} answer{'s' * (waiting != 1)} not yet confirmed]"
        print(line)

    # ---- speed and memory
    video = sum(r["video_seconds"] for r in results.values())
    took = sum(r["seconds"] for r in results.values())
    stage_totals = {s: round(sum(r["stages"].get(s, 0) for r in results.values()), 1) for s in STAGES}
    speed = {"video_seconds": round(video, 1), "seconds": round(took, 1), "stages": stage_totals,
             "peak_subprocess_mb": round(peak_child_mb())}
    print()
    line = f"speed: {video:.0f}s of video took {took:.0f}s ({video / took:.2f}x real time)" if took else "speed: -"
    if before and not partial and before.get("speed", {}).get("seconds"):
        b = before["speed"]
        line += f"; before: {b['seconds']:.0f}s ({b['video_seconds'] / b['seconds']:.2f}x)"
    print(line)
    print("  of which " + ", ".join(f"{s} {stage_totals[s]:.0f}s" for s in STAGES))
    print(f"memory: the largest subprocess peaked at {speed['peak_subprocess_mb']} MB")
    if before:
        moved = sum(1 for clip, r in results.items() if clip in before["clips"] and changes(before["clips"][clip], r))
        print(f"compared with '{args.against}' ({before.get('saved')}): {moved} of {len(results)} clips changed")
    if left_out:
        print("\nnot run:\n  " + "\n  ".join(left_out))

    if args.save:
        os.makedirs(RESULTS, exist_ok=True)
        with open(os.path.join(RESULTS, args.save + ".json"), "w", encoding="utf-8") as f:
            json.dump({"saved": time.strftime("%Y-%m-%d %H:%M"), "speed": speed, "clips": results}, f, indent=1)
        print(f"\nsaved to tests/eval/results/{args.save}.json")
    # MediaPipe can crash while Python shuts down; leave directly, as webapp/runner.py does.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if failed else 0)


if __name__ == "__main__":
    main()
