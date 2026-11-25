# FitCheck

A web app for AI-based fitness test assessment. Upload a video of a fitness test, and FitCheck tracks body pose with MediaPipe, counts reps, scores form, and produces an annotated video and a downloadable PDF report.

Built for Smart India Hackathon (SIH).

## Supported tests

Push-ups, squats, curl-ups, plank hold, jumping jacks, vertical jump, shuttle run, sit and reach, back stretch.

## Run locally

Requires Python 3.10.

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python webapp/app.py
```
## Project layout

```
FitCheck/
├── analysers/            one script per fitness test, run on each uploaded video
│   ├── pushup.py         squats.py, curl_ups.py, plank_hold.py, jumping_jack.py,
│   └── ...               vertical_jump.py, shuttle_run.py, sit_reach.py, back_stretch.py
├── tools/                standalone helpers, not used by the web app
│   ├── pose_viewer.py        general pose overlay on a video
│   ├── posture_analyzer.py   posture checks and suggestions
│   └── fitness_metrics.py    domain charts (strength, power, agility) from results
├── webapp/
│   ├── app.py            Flask server (port 5050)
│   ├── runner.py         runs an analyser headlessly for a job
│   ├── pdf_report.py     builds the downloadable PDF report
│   └── templates/        the single-page UI
└── requirements.txt
```

Each analyser can also be run on its own: `python analysers/pushup.py -v video.mp4`.

## Stack

Python, Flask, MediaPipe, OpenCV, NumPy, SciPy, Matplotlib, fpdf2
