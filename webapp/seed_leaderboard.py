"""Fill the leaderboard with sample scores for demos.

    .venv/bin/python webapp/seed_leaderboard.py          # add the sample scores
    .venv/bin/python webapp/seed_leaderboard.py --clear  # remove them again

Uses the same storage as the app (DATABASE_URL if set, else the SQLite file in FITCHECK_DATA_DIR or
webapp/). Sample entries are marked (their job id starts with "demo-"), so the page labels them
"Sample" and --clear removes only them, never real scores.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leaderboard  # noqa: E402

DATA_DIR = os.environ.get("FITCHECK_DATA_DIR") or os.path.dirname(os.path.abspath(__file__))
NAMES = ["Aarav", "Diya", "Kabir", "Ananya", "Vihaan", "Ishita", "Arjun", "Meera", "Rohan", "Saanvi K", "Neel", "Tara"]
RANGES = {"pushups": (12, 46), "plank": (35, 160), "jump": (24, 56), "squats60": (18, 44),
          "jacks30": (22, 41), "weekly": (15, 40)}

def seed(store, clear_only=False):
    """Replace the sample scores (never touches real ones)."""
    with store.lock, store._conn() as c:
        c.cursor().execute("DELETE FROM entries WHERE job LIKE 'demo-%'")
        c.commit()
    if clear_only:
        return
    rng = random.Random(7)
    for ch in leaderboard.CHALLENGES:
        r = leaderboard.resolve(ch)
        lo, hi = RANGES[r["id"]]
        for i, name in enumerate(rng.sample(NAMES, 8)):
            value = round(rng.uniform(lo, hi), 1) if r["unit"] == "s" else rng.randint(lo, hi)
            store.add(r["id"], r["period"], name, value, r["test"], f"demo-{r['id']}-{i}")


if __name__ == "__main__":
    seed(leaderboard.Store(DATA_DIR), clear_only="--clear" in sys.argv)
    print("Sample scores removed." if "--clear" in sys.argv else f"Added 8 sample scores to each of {len(leaderboard.CHALLENGES)} challenges.")
