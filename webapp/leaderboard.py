"""Challenges and their leaderboards.

Scores only come from analyses the server itself ran: when a job finishes, the server signs its
result (test, value, video length) into a token the browser keeps. Submitting a score sends that
token back, so a number can't be typed in or edited.

Storage: a hosted Postgres database when DATABASE_URL is set (survives restarts and deploys),
otherwise a SQLite file next to the jobs (fine locally; on Render's free plan it is wiped on
every deploy or restart).
"""
import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time

# kind: "best" = all-time best per person, "timed" = the video must be no longer than `seconds`,
# "weekly" = this week's featured challenge, rotating through `rotation`.
CHALLENGES = [
    {"id": "pushups", "kind": "best", "test": "pushup", "title": "Most push-ups", "unit": "reps",
     "blurb": "Valid push-ups in one video, any length."},
    {"id": "plank", "kind": "best", "test": "plank", "title": "Longest plank", "unit": "s",
     "blurb": "Hold a straight forearm plank as long as you can."},
    {"id": "jump", "kind": "best", "test": "verticaljump", "title": "Highest jump", "unit": "cm",
     "blurb": "Your best single vertical jump."},
    {"id": "squats60", "kind": "timed", "test": "squats", "seconds": 60, "title": "60-second squats", "unit": "reps",
     "blurb": "As many squats as you can in a video of 60 seconds or less."},
    {"id": "jacks30", "kind": "timed", "test": "jumpingjack", "seconds": 30, "title": "30-second jumping jacks",
     "unit": "reps", "blurb": "As many jumping jacks as you can in a video of 30 seconds or less."},
    {"id": "weekly", "kind": "weekly", "rotation": ["curlups", "pushup", "squats", "jumpingjack"],
     "title": "This week", "unit": "reps", "blurb": "A new test every Monday. The board resets each week."},
]
TEST_NAMES = {"pushup": "Push-ups", "squats": "Squats", "curlups": "Curl-ups", "plank": "Plank hold",
              "jumpingjack": "Jumping jacks", "verticaljump": "Vertical jump", "shuttlerun": "Shuttle run",
              "sitreach": "Sit and reach", "backstretch": "Back stretch"}
GRACE = 5            # seconds allowed over a timed challenge's limit (start and stop of the recording)
TOP = 50          # the page shows the top 5 and expands to these
NAME = re.compile(r"^[\w .'\-]{2,24}$", re.UNICODE)


def week_key(now=None):
    y, w, _ = (now or dt.date.today()).isocalendar()
    return f"{y}-W{w:02d}"


def resolve(ch, now=None):
    """The challenge as it stands now (weekly ones pick this week's test)."""
    ch = dict(ch)
    if ch["kind"] == "weekly":
        y, w, _ = (now or dt.date.today()).isocalendar()
        ch["test"] = ch["rotation"][(y * 53 + w) % len(ch["rotation"])]
        ch["title"] = f"This week: most {TEST_NAMES[ch['test']].lower()}"
        ch["period"] = week_key(now)
    else:
        ch["period"] = "all"
    ch["test_name"] = TEST_NAMES[ch["test"]]
    return ch


# --------------------------------------------------------------- tokens -----

def _secret(data_dir):
    s = os.environ.get("FITCHECK_SECRET")
    if s:
        return s.encode()
    path = os.path.join(data_dir, "leaderboard.secret")      # generated once, kept beside the data
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
    with open(path) as f:
        return f.read().strip().encode()


def sign(data_dir, payload):
    body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    mac = hmac.new(_secret(data_dir), body.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{body}.{mac}"


def verify(data_dir, token):
    try:
        body, mac = token.rsplit(".", 1)
        if not hmac.compare_digest(mac, hmac.new(_secret(data_dir), body.encode(), hashlib.sha256).hexdigest()[:32]):
            return None
        return json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except Exception:
        return None


# --------------------------------------------------------------- storage ----

class Store:
    def __init__(self, data_dir):
        self.url = os.environ.get("DATABASE_URL")
        self.path = os.path.join(data_dir, "leaderboard.db")
        self.lock = threading.Lock()
        self.ph = "%s" if self.url else "?"
        with self.lock, self._conn() as c:
            cur = c.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS entries (
                challenge TEXT NOT NULL, period TEXT NOT NULL, name TEXT NOT NULL, value REAL NOT NULL,
                test_key TEXT NOT NULL, created REAL NOT NULL, job TEXT NOT NULL,
                UNIQUE (challenge, period, job))""")
            c.commit()

    def _conn(self):
        if self.url:
            import psycopg                       # only needed when a hosted database is configured
            return psycopg.connect(self.url)
        return sqlite3.connect(self.path)

    def add(self, challenge, period, name, value, test_key, job):
        q = (f"INSERT INTO entries (challenge, period, name, value, test_key, created, job) "
             f"VALUES ({', '.join([self.ph] * 7)})")
        with self.lock, self._conn() as c:
            try:
                c.cursor().execute(q, (challenge, period, name, value, test_key, time.time(), job))
                c.commit()
                return True
            except Exception as e:               # the same analysis submitted twice
                if "unique" in str(e).lower() or "duplicate" in str(e).lower():
                    return False
                raise

    def top(self, challenge, period, n=TOP):
        """Each person's best entry, best first."""
        # The pattern goes in as a parameter: a literal % in the SQL breaks Postgres's driver.
        q = (f"SELECT name, MAX(value) AS best, MIN(created), MIN(CASE WHEN job LIKE {self.ph} THEN 1 ELSE 0 END) "
             f"FROM entries WHERE challenge = {self.ph} AND period = {self.ph} "
             f"GROUP BY name ORDER BY best DESC, MIN(created) ASC LIMIT {int(n)}")
        with self.lock, self._conn() as c:
            cur = c.cursor(); cur.execute(q, ("demo-%", challenge, period))
            # sample: only sample scores (seed_leaderboard.py) behind this row, so the page can say so
            return [{"name": r[0], "value": r[1], "created": r[2], "sample": bool(r[3])} for r in cur.fetchall()]

    def best(self, challenge, period, name):
        q = f"SELECT MAX(value) FROM entries WHERE challenge = {self.ph} AND period = {self.ph} AND name = {self.ph}"
        with self.lock, self._conn() as c:
            cur = c.cursor(); cur.execute(q, (challenge, period, name))
            return cur.fetchone()[0]

    def rank(self, challenge, period, name):
        q = (f"SELECT COUNT(*) FROM (SELECT name, MAX(value) AS best FROM entries WHERE challenge = {self.ph} AND period = {self.ph} "
             f"GROUP BY name) t WHERE best > (SELECT MAX(value) FROM entries WHERE challenge = {self.ph} AND period = {self.ph} AND name = {self.ph})")
        with self.lock, self._conn() as c:
            cur = c.cursor(); cur.execute(q, (challenge, period, challenge, period, name))
            return cur.fetchone()[0] + 1


def ends_at(now=None):
    """When this week's challenge ends: next Monday 00:00 UTC, as a Unix time."""
    now = now or dt.datetime.now(dt.timezone.utc)
    monday = (now - dt.timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    return (monday + dt.timedelta(days=7)).timestamp()


def board(store, name=None):
    """Every challenge with its ranking; with a name, also that person's best and rank in each."""
    name = " ".join((name or "").split())
    out = []
    for ch in CHALLENGES:
        r = resolve(ch)
        item = {k: r.get(k) for k in ("id", "kind", "title", "blurb", "unit", "test", "test_name", "seconds", "period")}
        item["entries"] = store.top(r["id"], r["period"])
        if r["kind"] == "weekly":
            item["ends"] = ends_at()
        if name and NAME.match(name):
            best = store.best(r["id"], r["period"], name)
            item["you"] = {"name": name, "value": best, "rank": store.rank(r["id"], r["period"], name)} if best is not None else None
        out.append(item)
    return out


def submit(store, data_dir, token, challenge_id, name):
    """Returns (ok, message, rank)."""
    claim = verify(data_dir, token or "")
    if not claim:
        return False, "This result can't be submitted (it wasn't analysed by FitCheck, or it is too old).", None
    ch = next((c for c in CHALLENGES if c["id"] == challenge_id), None)
    if not ch:
        return False, "That challenge doesn't exist.", None
    ch = resolve(ch)
    name = " ".join((name or "").split())
    if not NAME.match(name):
        return False, "Use a display name of 2 to 24 letters, numbers or spaces.", None
    if claim.get("test") != ch["test"]:
        return False, f"This challenge needs a {ch['test_name'].lower()} analysis.", None
    if ch["kind"] == "timed" and (claim.get("seconds") or 1e9) > ch["seconds"] + GRACE:
        return False, f"For this challenge the video must be {ch['seconds']} seconds or shorter.", None
    if ch["kind"] == "weekly" and claim.get("week") != ch["period"]:
        return False, "This analysis was done before this week's challenge started.", None
    if float(claim.get("value") or 0) <= 0:
        return False, "Nothing was counted in this video, so there's no score to enter.", None
    if not store.add(ch["id"], ch["period"], name, float(claim["value"]), claim["test"], claim["job"]):
        return True, "Already on the board.", store.rank(ch["id"], ch["period"], name)   # same analysis again: fine
    return True, "Submitted.", store.rank(ch["id"], ch["period"], name)
