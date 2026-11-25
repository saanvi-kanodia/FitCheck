FROM python:3.11-slim

# OpenCV (pulled in by MediaPipe) needs these system libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# One worker: jobs are tracked in memory, so every request must reach the same process
CMD gunicorn --chdir webapp --workers 1 --threads 4 --timeout 600 --bind 0.0.0.0:${PORT:-10000} app:app
