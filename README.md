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

- `webapp/`: Flask app (`app.py`), job runner, PDF report builder, and the HTML template
- `*_analysis.py`: one analyser per fitness test, run on each uploaded video
- `posture_analyzer.py`, `metric.py`, `gym_code.py`: shared pose and metric helpers

## Stack

Python, Flask, MediaPipe, OpenCV, NumPy, SciPy, Matplotlib, fpdf2
