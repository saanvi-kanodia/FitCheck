# Accuracy and speed checks

`run.py` sends every clip listed in `manifest.json` through the same steps as a real upload
(shrink, body check, analyser, encode, PDF) and compares what comes out with an answer a
person worked out by watching the clip. It also reports how long each step took and how much
memory the heaviest step used.

```
.venv/bin/python tests/eval/run.py                      # run everything
.venv/bin/python tests/eval/run.py --test squats        # one test
.venv/bin/python tests/eval/run.py --clip plank-1       # clips whose path contains this
.venv/bin/python tests/eval/run.py --save baseline      # keep the results (full runs only)
.venv/bin/python tests/eval/run.py --against baseline   # show what changed since then
.venv/bin/python tests/eval/run.py --fast               # quick re-test (see below)
```

Run it before and after any change to an analyser. A change to one test should leave the
rows for the other eight unchanged; `--against` lists every clip whose result moved.

It works in a temporary folder and never touches `webapp/jobs`.

`--fast` keeps each clip's pose tracking in `tests/eval/cache/` the first time and reuses it
afterwards, so a re-test skips the pose model. The results are exactly the same as a normal
run; the timings are not, so use a normal run to measure speed. The cache is only reused for
the same video and the same pose settings, and it can be deleted at any time.

The four tests with no clip of their own yet (squats, curl-ups, sit and reach, back stretch)
are each run on a clip of another exercise. Those rows are marked `smoke`: there is no right
answer, they only show a crash or a result that changed when it should not have.

The quick checks of the shared pose code (no videos needed) are separate:

```
.venv/bin/python -m unittest discover -s tests/unit
```

Setting up on an Intel Mac: `pip install --only-binary=:all: -r requirements.txt pillow`
(without `--only-binary`, pip tries to compile `sentencepiece` and fails).

## Reading the table

| column | meaning |
|---|---|
| expected / got | the answer for the clip and what the app produced. `refuse` / `refused` means the app turned the video away before analysing it |
| result | `ok` if within the tolerance, `WRONG` (with how far off) if not, `-` if the clip has no answer |
| other figures | further numbers from the report that have an answer, such as reps attempted |
| form problems | `ok`, or how many form problems were missed or flagged without reason |
| answer | `confirmed` once a person has checked the answer, `proposed` until then, `smoke` for a run with no right answer |
| time | processing time, and video length divided by it (above 1x is faster than real time) |

Only clips with a confirmed answer, and crashes, make the command exit with an error.

## Adding a clip

1. Put the video in `tests/eval/clips/` (this folder is not committed: the clips are
   personal videos and are large).
2. Add an entry to `manifest.json`:

| field | meaning |
|---|---|
| `clip` | path from the project root |
| `test` | `pushup`, `squats`, `curlups`, `plank`, `jumpingjack`, `verticaljump`, `shuttlerun`, `sitreach`, `backstretch` |
| `expected` | the true answer, in the unit the app shows: reps, seconds (plank), cm (jump, reach, back stretch), m/s (shuttle run) |
| `tolerance` | how far off still counts as correct. `0` for rep counts |
| `expected_extra` | optional: other figures from the report by their label, e.g. `{"Push-ups attempted": 12}` for 10 full reps plus 2 incomplete ones |
| `expected_issues` | optional: the form problems a person can see in the clip, by the name the report uses, e.g. `["Hips sagging"]`. `[]` means clean form: nothing should be flagged |
| `expect_status` | optional: `rejected` for clips the app should refuse (body out of frame, wrong camera angle) |
| `verified` | `true` once a person has watched the clip and confirmed the answers |
| `height` | sit and reach only: the person's height in cm |
| `skip` | optional: a reason. The clip is left out unless it is named with `--clip` |
| `smoke` | optional: `true` for a clip of another exercise, run only to catch crashes and changes |
| `notes` | camera angle, lighting, which way the person faces, when each rep happens, anything unusual |

A good set for each test has 3 to 5 clips covering: clean reps, some deliberately incomplete
reps, the person facing each direction, dim lighting, and one awkward camera angle.
