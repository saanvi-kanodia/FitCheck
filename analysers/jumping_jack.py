import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description='Jumping Jack Analysis.')
    parser.add_argument('-v', '--video', type=str, default='jumpingjacks.mp4', help='Input video file')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output analyzed video file')
    parser.add_argument('--det', type=float, default=0.5, help='Detection confidence')
    parser.add_argument('--track', type=float, default=0.5, help='Tracking confidence')
    parser.add_argument('-c', '--complexity', type=int, default=0, help='Model complexity (0, 1, 2)')
    return parser.parse_args()

def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def main():
    args = parse_args()

    mp_pose = mp.solutions.pose

    vid = cv2.VideoCapture(args.video)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    if args.output is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        input_name = args.video.split('.')[0] if args.video else 'jumpingjacks'
        os.makedirs("analysed", exist_ok=True)
        args.output = f"analysed/{input_name}_analyzed_{timestamp}.mp4"

    print(f"📹 Saving analyzed video as: {args.output}")

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, codec, fps, (width, height))

    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=args.det, min_tracking_confidence=args.track)

    rep_count = 0
    in_up_phase = False
    last_rep_time = None

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            head = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]

            # Detect up-phase
            arms_up = left_wrist[1] < head[1] and right_wrist[1] < head[1]
            legs_apart = calculate_distance(left_ankle, right_ankle) > calculate_distance(left_hip, right_hip) * 1.5

            # Detect down-phase
            arms_down = left_wrist[1] > left_hip[1] and right_wrist[1] > right_hip[1]
            legs_together = calculate_distance(left_ankle, right_ankle) < calculate_distance(left_hip, right_hip) * 1.2

            if arms_up and legs_apart:
                in_up_phase = True

            if in_up_phase and arms_down and legs_together:
                current_time = time.time()
                if last_rep_time and current_time - last_rep_time < 0.5:  # Momentum-based cheating
                    cv2.putText(frame, "Cheat: Too fast", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    rep_count += 1
                    last_rep_time = current_time
                in_up_phase = False

            # Cheat detection: Lateral torso shift
            torso_shift = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x)
            if torso_shift > 0.1:  # Threshold for lateral shift
                cv2.putText(frame, "Cheat: Lateral shift", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(frame, f"Reps: {rep_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        out.write(frame)
        cv2.imshow('Jumping Jack Analysis', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Total Reps: {rep_count}")

if __name__ == "__main__":
    main()
