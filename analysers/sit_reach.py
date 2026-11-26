import cv2
import mediapipe as mp
from core.pose import Tracker
import numpy as np
import argparse
import time
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description='Sit-and-Reach Analysis.')
    parser.add_argument('-v', '--video', type=str, default='sitreach.mp4', help='Input video file')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output analyzed video file')
    parser.add_argument('--height', type=float, required=True, help='User-reported height in cm')
    parser.add_argument('--det', type=float, default=0.5, help='Detection confidence')
    parser.add_argument('--track', type=float, default=0.5, help='Tracking confidence')
    parser.add_argument('-c', '--complexity', type=int, default=0, help='Model complexity (0, 1, 2)')
    return parser.parse_args()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def calibrate_scale(hip, ankle, user_height):
    leg_length_px = np.linalg.norm(np.array(hip) - np.array(ankle))
    leg_length_cm = user_height * 0.53  # Approximate leg length as 53% of height
    return leg_length_cm / leg_length_px

def main():
    args = parse_args()

    mp_pose = mp.solutions.pose

    vid = cv2.VideoCapture(args.video)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS) or 30   # the true rate, so the saved video keeps time with the original

    if args.output is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        input_name = args.video.split('.')[0] if args.video else 'sitreach'
        os.makedirs("analysed", exist_ok=True)
        args.output = f"analysed/{input_name}_analyzed_{timestamp}.mp4"

    print(f"📹 Saving analyzed video as: {args.output}")

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, codec, fps, (width, height))

    pose = Tracker(video=args.video, static_image_mode=False, min_detection_confidence=args.det, min_tracking_confidence=args.track)

    scale = None
    max_reach_cm = 0

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            knee_angle = calculate_angle(hip, knee, ankle)

            if scale is None and knee_angle >= 175:  # Calibrate scale when legs are straight
                scale = calibrate_scale(hip, ankle, args.height)

            if scale and knee_angle >= 175:  # Ensure legs are straight
                reach_px = wrist[0] - ankle[0]
                reach_cm = reach_px * scale
                max_reach_cm = max(max_reach_cm, reach_cm)

                cv2.putText(frame, f"Reach: {reach_cm:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            if knee_angle < 170:
                cv2.putText(frame, "Cheat: Bent legs detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(frame)
        cv2.imshow('Sit-and-Reach Analysis', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Maximum Reach: {max_reach_cm:.2f} cm")

if __name__ == "__main__":
    main()
