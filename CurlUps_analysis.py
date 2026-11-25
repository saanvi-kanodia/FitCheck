import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description='Curl-Ups Analysis.')
    parser.add_argument('-v', '--video', type=str, default='curlups.mp4', help='Input video file')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output analyzed video file')
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

def main():
    args = parse_args()

    mp_pose = mp.solutions.pose

    vid = cv2.VideoCapture(args.video)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    if args.output is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        input_name = args.video.split('.')[0] if args.video else 'curlups'
        os.makedirs("analysed", exist_ok=True)
        args.output = f"analysed/{input_name}_analyzed_{timestamp}.mp4"

    print(f"📹 Saving analyzed video as: {args.output}")

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, codec, fps, (width, height))

    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=args.det, min_tracking_confidence=args.track)

    rep_count = 0
    in_curl = False
    start_time = time.time()

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            torso_angle = calculate_angle(shoulder, hip, knee)

            if torso_angle <= 140 and not in_curl:  # Valid curl detected
                in_curl = True

            if torso_angle >= 175 and in_curl:  # Return to baseline
                in_curl = False
                rep_count += 1

            cv2.putText(frame, f"Torso Angle: {torso_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Rep Count: {rep_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        elapsed_time = time.time() - start_time
        if elapsed_time > 60:  # Stop after 60 seconds
            break

        out.write(frame)
        cv2.imshow('Curl-Ups Analysis', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Total Reps: {rep_count}")

if __name__ == "__main__":
    main()
