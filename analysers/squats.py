import cv2
import mediapipe as mp
from core.pose import Tracker
import numpy as np
import argparse
import time
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description='Squats Analysis.')
    parser.add_argument('-v', '--video', type=str, default='squats.mp4', help='Input video file')
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
    fps = vid.get(cv2.CAP_PROP_FPS) or 30   # the true rate, so the saved video keeps time with the original

    if args.output is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        input_name = args.video.split('.')[0] if args.video else 'squats'
        os.makedirs("analysed", exist_ok=True)
        args.output = f"analysed/{input_name}_analyzed_{timestamp}.mp4"

    print(f"📹 Saving analyzed video as: {args.output}")

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, codec, fps, (width, height))

    pose = Tracker(video=args.video, static_image_mode=False, min_detection_confidence=args.det, min_tracking_confidence=args.track)

    squat_count = 0
    is_down = False
    start_time = time.time()

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        elapsed_time = time.time() - start_time
        if elapsed_time > 60:  # Stop after 60 seconds
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

            # Cheat detection: Depth not reached
            if min(left_knee_angle, right_knee_angle) > 100:
                cv2.putText(frame, "Cheat: Depth not reached", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Cheat detection: Asymmetric hip height
            hip_height_diff = abs(left_hip[1] - right_hip[1])
            if hip_height_diff > 0.1:  # 10% difference
                cv2.putText(frame, "Cheat: Asymmetric hip height", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Cheat detection: Knee drift
            left_knee_drift = abs(left_knee[0] - left_ankle[0])
            right_knee_drift = abs(right_knee[0] - right_ankle[0])
            if left_knee_drift > 0.15 or right_knee_drift > 0.15:  # 15% drift
                cv2.putText(frame, "Cheat: Knee drift detected", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Squat counting logic
            if min(left_knee_angle, right_knee_angle) <= 90 and not is_down:  # Down phase detected
                is_down = True

            if max(left_knee_angle, right_knee_angle) >= 160 and is_down:  # Up phase detected
                squat_count += 1
                is_down = False

            cv2.putText(frame, f"Squats: {squat_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        out.write(frame)
        cv2.imshow('Squats Analysis', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Total Squats: {squat_count}")

if __name__ == "__main__":
    main()
