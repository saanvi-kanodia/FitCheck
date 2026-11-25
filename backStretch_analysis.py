import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description='Back Stretch Test Analysis.')
    parser.add_argument('-v', '--video', type=str, default='backstretch.mp4', help='Input video file')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output analyzed video file')
    parser.add_argument('--det', type=float, default=0.5, help='Detection confidence')
    parser.add_argument('--track', type=float, default=0.5, help='Tracking confidence')
    parser.add_argument('-c', '--complexity', type=int, default=0, help='Model complexity (0, 1, 2)')
    return parser.parse_args()

def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.abs(a[1] - b[1])  # Distance in y-axis

def calculate_angle(a, b):
    a = np.array(a)
    b = np.array(b)

    radians = np.arctan2(b[1] - a[1], b[0] - a[0])
    angle = np.abs(radians * 180.0 / np.pi)

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
        input_name = args.video.split('.')[0] if args.video else 'backstretch'
        os.makedirs("analysed", exist_ok=True)
        args.output = f"analysed/{input_name}_analyzed_{timestamp}.mp4"

    print(f"📹 Saving analyzed video as: {args.output}")

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, codec, fps, (width, height))

    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=args.det, min_tracking_confidence=args.track)

    upper_arm_length = None
    fingertip_distance_cm = None

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Cheat detection: Missing landmarks
            if not all([left_wrist, right_wrist]):
                cv2.putText(frame, "Cheat: Missing landmarks", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                out.write(frame)
                continue

            # Cheat detection: Wrong camera orientation
            if left_wrist[1] > height or right_wrist[1] > height:
                cv2.putText(frame, "Cheat: Hands not visible in frame", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                out.write(frame)
                continue

            # Cheat detection: Torso twist
            shoulder_angle = calculate_angle(left_shoulder, right_shoulder)
            if shoulder_angle > 15:
                cv2.putText(frame, "Cheat: Torso twist detected", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Measure distance between wrists in y-axis
            wrist_distance_px = calculate_distance(left_wrist, right_wrist)

            # Calibrate upper-arm length if not already done
            if upper_arm_length is None:
                upper_arm_length = calculate_distance(left_shoulder, left_elbow)

            # Convert px to cm
            if upper_arm_length:
                fingertip_distance_cm = (wrist_distance_px / upper_arm_length) * 30  # Approximate upper-arm length in cm

            # Display results
            if fingertip_distance_cm is not None:
                score_text = f"Score: {fingertip_distance_cm:.2f} cm"
                cv2.putText(frame, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        out.write(frame)
        cv2.imshow('Back Stretch Test Analysis', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()

    if fingertip_distance_cm is not None:
        print(f"Final Score: {fingertip_distance_cm:.2f} cm")

if __name__ == "__main__":
    main()
