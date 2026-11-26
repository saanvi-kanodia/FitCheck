# FitCheck

AI-based fitness test assessment in the browser. Upload a video of a fitness test and FitCheck tracks body pose with MediaPipe, counts reps, scores form, and gives back an annotated video and a downloadable PDF report.

**Try it live:** [fitcheck-anue.onrender.com](https://fitcheck-anue.onrender.com/)
(It runs on Render's free tier, so the first load after a quiet spell can take a minute while the server wakes up.)

Built for Smart India Hackathon (SIH).

## Supported tests

| Test | What it measures |
| --- | --- |
| Push-ups | Rep count and form |
| Squats | Rep count and depth |
| Curl-ups | Rep count |
| Plank hold | Hold time and body alignment |
| Jumping jacks | Rep count |
| Vertical jump | Jump height |
| Shuttle run | Run time |
| Sit and reach | Flexibility (asks for your height in cm) |
| Back stretch | Flexibility |

## How it works

1. **Upload.** The browser sends the video in 2 MB chunks, up to 50 MB in total.
2. **Prepare.** The server uses ffmpeg to scale the video down to 640 px on its longest side.
3. **Analyse.** The test's script in `analysers/` runs headlessly through `webapp/runner.py`. This wrapper:
   - saves preview frames and progress instead of opening a window
   - times everything by the video's own frame rate, so results don't depend on server speed
4. **Report.** The app shows the annotated video and results, and `pdf_report.py` builds a PDF summary.

Jobs and uploads are deleted from the server after 24 hours. Your history stays in your browser.

## Live camera mode

Every test can also be done live. Pick a test and press **Analyse [test] live with camera**.

- Pose tracking runs entirely in your browser (MediaPipe Tasks), so the camera video never leaves your device.
- A 3-second countdown gives you time to get into position before anything is counted.
- The overlay shows a live count or measurement:
  - reps for push-ups, squats, curl-ups and jumping jacks
  - seconds for the plank
  - jumps and laps for vertical jump and shuttle run
  - best reach and best forward bend for sit and reach and back stretch
- Cues on screen say when a rep wasn't counted and why, or when you've moved out of frame.
- Pressing **Stop and save** keeps the recording, results and a PDF in your analyses list.
- Enter your height to get vertical jump and sit-and-reach results in centimetres. Without it they are shown relative to leg length.

Live results are estimates and less precise than an uploaded analysis. Live mode needs HTTPS (or localhost) and a browser that can record video, such as Chrome, Edge or Safari.

## Project structure

```
analysers/      one MediaPipe + OpenCV script per fitness test
webapp/
  app.py        Flask app: chunked uploads, job queue, results, PDF download
  runner.py     runs an analyser headlessly and reports progress
  pdf_report.py builds the PDF report
  body_check.py, overlay_text.py   helpers for input checks and on-video text
  templates/    the single-page UI
  static/       sample videos you can try
tools/          standalone scripts for pose viewing, posture analysis and fitness metrics
Dockerfile      container used for deployment
render.yaml     Render deployment config
```

## Run locally

Requires Python 3.11.

```bash
git clone https://github.com/saanvi-kanodia/FitCheck.git
cd FitCheck
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python webapp/app.py
```

Then open http://localhost:5050.

To run one analyser on its own, outside the web app:

```bash
python analysers/squats.py -v path/to/video.mp4 -o out.mp4
python analysers/sit_reach.py -v path/to/video.mp4 --height 165
```

### With Docker

```bash
docker build -t fitcheck .
docker run -p 10000:10000 fitcheck
```

Then open http://localhost:10000. The container runs gunicorn with a single worker on purpose, because jobs are tracked in memory and every request has to reach the same process.

## Filming tips

Keep clips under a minute and use a steady camera. The app also shows these tips for each test:

| Test | How to film |
| --- | --- |
| Push-ups | From the side at floor level, whole body from head to heels in frame |
| Squats | From the side, whole body in frame including your feet |
| Curl-ups | From the side, knees bent and feet flat on the floor |
| Plank hold | From the side at floor level, so shoulders, hips and ankles are visible |
| Jumping jacks | From the front, with room above your hands and below your feet |
| Vertical jump | From the side, feet in frame for the whole jump |
| Shuttle run | From the side, far enough back to see both turning markers |
| Sit and reach | From the side, legs straight, reaching forward slowly |
| Back stretch | From the side, bending forward slowly from the hips |

## Tech stack

Python, Flask, MediaPipe, OpenCV, NumPy, SciPy, Matplotlib, fpdf2, ffmpeg, gunicorn, Docker, Render
