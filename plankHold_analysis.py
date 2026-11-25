import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description='Plank Hold Analysis.')
    parser.add_argument('-v', '--video', type=str, default='plank.mp4', help='Input video file')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output analyzed video file')
    parser.add_argument('--det', type=float, default=0.5, help='Detection confidence')
    parser.add_argument('--track', type=float, default=0.5, help='Tracking confidence')
    parser.add_argument('-c', '--complexity', type=int, default=0, help='Model complexity (0, 1, 2)')
    return parser.parse_args()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
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
        input_name = args.video.split('.')[0] if args.video else 'plank'
        os.makedirs("analysed", exist_ok=True)
        args.output = f"analysed/{input_name}analyzed{timestamp}.mp4"

    print(f"📹 Saving analyzed video as: {args.output}")

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, codec, fps, (width, height))

    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=args.det, min_tracking_confidence=args.track)

    initial_time = time.time()
    start_time = None
    valid_plank = True
    cheat_start_time = None
    total_duration = 0
    cheat_events = []
    form_wrong_times = []
    last_cheat_time = {'sagging': 0.0, 'piked': 0.0, 'bent_knees': 0.0, 'arched_neck': 0.0, 'lateral': 0.0}
    good_form_counter = 0  # Counter for consecutive good form frames to start plank

    # Analytics tracking
    session_analytics = {
        'total_frames': 0,
        'good_form_frames': 0,
        'bad_form_frames': 0,
        'form_scores': [],
        'session_start_time': time.time()
    }

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
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            # Calculate key angles
            shoulder_hip_ankle_angle = calculate_angle(shoulder, hip, ankle)
            ear_shoulder_hip_angle = calculate_angle(left_ear, shoulder, hip)
            hip_knee_ankle_angle = calculate_angle(hip, left_knee, ankle)
            left_arm_angle = calculate_angle(shoulder, elbow, wrist)
            right_arm_angle = calculate_angle(shoulder_r, elbow_r, wrist_r)
            left_leg_angle = calculate_angle(left_hip, left_knee, ankle)
            right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)

            # FORM SCORING SYSTEM (out of 4)
            form_score = 0
            
            # 1. Spine/Hip Alignment (170-180°)
            if 170 <= shoulder_hip_ankle_angle <= 180:
                form_score += 1
            
            # 2. Head/Neck Alignment (170-180°)
            if 170 <= ear_shoulder_hip_angle <= 180:
                form_score += 1
            
            # 3. Leg Alignment (170-180°)
            if 170 <= hip_knee_ankle_angle <= 180:
                form_score += 1
            
            # 4. Symmetry (arm and leg angles within 30°)
            if abs(left_arm_angle - right_arm_angle) < 30 and abs(left_leg_angle - right_leg_angle) < 30:
                form_score += 1
            
            # Track analytics (only during active plank for form scores)
            session_analytics['total_frames'] += 1
            if start_time is not None:
                session_analytics['form_scores'].append(form_score)
                if form_score >= 3:
                    session_analytics['good_form_frames'] += 1
                else:
                    session_analytics['bad_form_frames'] += 1
            else:
                session_analytics['bad_form_frames'] += 1  # Frames before plank start are not good

            # Check for sustained good form to start plank
            if form_score >= 3 and 170 <= shoulder_hip_ankle_angle <= 180 and shoulder[1] < hip[1]:
                good_form_counter += 1
            else:
                good_form_counter = 0

            if good_form_counter >= 10 and start_time is None:  # Require 10 consecutive good frames to start plank
                start_time = time.time()
                valid_plank = True
                cheat_start_time = None
            elif shoulder_hip_ankle_angle < 165 or shoulder_hip_ankle_angle > 185:  # Invalid plank
                if valid_plank:
                    cheat_start_time = time.time()
                    if start_time is not None:
                        form_wrong_times.append(time.time() - initial_time)
                    valid_plank = False
                elif cheat_start_time and time.time() - cheat_start_time > 2:  # Timer stops after 2 seconds of invalid plank
                    if start_time is not None:
                        total_duration += time.time() - start_time
                    start_time = None

            # Cheat detection (only during active plank)
            if start_time is not None:
                if shoulder_hip_ankle_angle < 165:  # Sagging hips
                    cv2.putText(frame, "Cheat: Sagging hips", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if time.time() - last_cheat_time['sagging'] > 1:
                        cheat_events.append({'time': time.time() - initial_time, 'type': 'sagging'})
                        last_cheat_time['sagging'] = time.time()

                if shoulder_hip_ankle_angle > 185:  # Piked hips
                    cv2.putText(frame, "Cheat: Piked hips", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if time.time() - last_cheat_time['piked'] > 1:
                        cheat_events.append({'time': time.time() - initial_time, 'type': 'piked'})
                        last_cheat_time['piked'] = time.time()

                if hip_knee_ankle_angle < 170 or right_leg_angle < 170:  # Bent knees
                    cv2.putText(frame, "Cheat: Bent knees", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if time.time() - last_cheat_time['bent_knees'] > 1:
                        cheat_events.append({'time': time.time() - initial_time, 'type': 'bent knees'})
                        last_cheat_time['bent_knees'] = time.time()

                if ear_shoulder_hip_angle < 165:  # Arched neck
                    cv2.putText(frame, "Cheat: Arched neck", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if time.time() - last_cheat_time['arched_neck'] > 1:
                        cheat_events.append({'time': time.time() - initial_time, 'type': 'arched neck'})
                        last_cheat_time['arched_neck'] = time.time()

                shoulder_x_variance = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x)
                if shoulder_x_variance * width > 15:  # Excessive lateral shift
                    cv2.putText(frame, "Cheat: Lateral shift", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if time.time() - last_cheat_time['lateral'] > 1:
                        cheat_events.append({'time': time.time() - initial_time, 'type': 'lateral shift'})
                        last_cheat_time['lateral'] = time.time()

            if start_time:
                elapsed_time = time.time() - start_time
                cv2.putText(frame, f"Duration: {elapsed_time:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Form Score: {form_score}/4", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        out.write(frame)
        cv2.imshow('Plank Hold Analysis', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    if start_time is not None:
        total_duration += time.time() - start_time

    vid.release()
    out.release()
    cv2.destroyAllWindows()

    total_video_time = time.time() - initial_time
    avg_form_score = np.mean(session_analytics['form_scores']) if session_analytics['form_scores'] else 0
    form_consistency = (session_analytics['good_form_frames'] / len(session_analytics['form_scores']) * 100) if session_analytics['form_scores'] else 0

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    downloads_dir = os.path.expanduser('~/Downloads')
    os.makedirs(downloads_dir, exist_ok=True)
    report_file = os.path.join(downloads_dir, f"plank_report_{timestamp}.txt")

    report = f"""
Exercise: plank_hold
Value: {total_duration:.2f} 
Form_Score: {avg_form_score:.1f}
Max_Form_Score: 4

Session Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Original Video: {args.video}
Analyzed Video: {args.output}
Session Duration: {total_video_time:.1f} seconds

PERFORMANCE SUMMARY:
================================================================================
Total Plank Hold Duration: {total_duration:.2f} seconds
Average Form Score: {avg_form_score:.1f}/4.0

DETAILED ANALYTICS:
================================================================================
Total Frames Analyzed: {session_analytics['total_frames']}
Good Form Frames: {session_analytics['good_form_frames']}
Poor Form Frames: {len(session_analytics['form_scores']) - session_analytics['good_form_frames']}

CHEATS DETECTED:
================================================================================
"""
    for cheat in cheat_events:
        report += f"- {cheat['time']:.2f}s: {cheat['type']}\n"

    report += f"""
FORM WRONG TIMESTAMPS:
================================================================================
"""
    for t in form_wrong_times:
        report += f"- {t:.2f}s\n"

    report += f"""
================================================================================
Report generated by Fitness Tracking App v2.0
================================================================================
"""

    with open(report_file, 'w') as f:
        f.write(report)

    print(f"Total Plank Hold Duration: {total_duration:.2f} seconds")
    print(f"Average Form Score: {avg_form_score:.2f}/4.0")
    print(f"Report saved to: {report_file}")

if __name__ == "__main__":
    main()