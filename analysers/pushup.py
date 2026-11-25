import sys
import cv2
import mediapipe as mp
import numpy as np
import argparse
import numpy as np
import time
import os
import json

def parse_args():
    parser = argparse.ArgumentParser(
        description='Testing for Handstand.')
    parser.add_argument('-v','--video', type=str, default = 'Squats.mp4')
    parser.add_argument('-o','--output', type=str, default =None)
    parser.add_argument('--det',type=float, default = 0.5,help='Detection confidence')
    parser.add_argument('--track',type=float, default =0.5,help='Tracking confidence')
    parser.add_argument('-wt','--workout_type',type=str, default ='PushUp')
    parser.add_argument('-c','--complexity', type=int, default = 0,help='Complexity of the model options 0,1,2')
    
    return parser.parse_args()


args=parse_args()

line_color = (255,255,255)
# def estimateSpeed(location1, location2):
# 	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
# 	# ppm = location2[2] / carWidht
# 	ppm = 8.8
# 	d_meters = d_pixels / ppm
# 	#print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
# 	fps = 18
# 	speed = d_meters * fps * 3.6
# 	return speed

start_time = time.time()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians *180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle

def generate_analytics_report(analytics, video_filename, output_video=None):
    """Generate a comprehensive analytics report"""
    session_duration = time.time() - analytics['session_start_time']
    
    # Calculate statistics
    avg_form_score = np.mean(analytics['form_scores']) if analytics['form_scores'] else 0
    avg_arm_angle = np.mean(analytics['avg_arm_angles']) if analytics['avg_arm_angles'] else 0
    form_consistency = (analytics['good_form_frames'] / analytics['total_frames'] * 100) if analytics['total_frames'] > 0 else 0
    analytics['attempted_pushups'] = max(analytics['attempted_pushups'], analytics['valid_pushups'])
    success_rate = (analytics['valid_pushups'] / analytics['attempted_pushups'] * 100) if analytics['attempted_pushups'] > 0 else 0
    
    # Create report content
    report = f"""

Exercise: pushups
Value: {analytics['valid_pushups']} 
Form_Score: {avg_form_score:.1f}
Max_Form_Score: 4
Session Duration: {session_duration:.1f} 

Session Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Original Video: {video_filename}
Analyzed Video: {output_video if output_video else 'Not saved'}

PERFORMANCE SUMMARY:
================================================================================
Valid Pushups Completed: {analytics['valid_pushups']}
Total Pushup Attempts: {analytics['attempted_pushups']}
Success Rate: {success_rate:.1f}%
Average Form Score: {avg_form_score:.1f}/4.0
Form Consistency: {form_consistency:.1f}% good form

DETAILED ANALYTICS:
================================================================================
Total Frames Analyzed: {analytics['total_frames']}
Good Form Frames: {analytics['good_form_frames']}
Poor Form Frames: {analytics['bad_form_frames']}
Average Arm Angle: {avg_arm_angle:.1f}°

FORM VIOLATIONS BREAKDOWN:
================================================================================
Arm Asymmetry Issues: {analytics['symmetry_violations']}
Body Alignment Issues: {analytics['body_alignment_violations']}
Leg Position Issues: {analytics['leg_position_violations']}
Head Position Issues: {analytics['head_position_violations']}

RECOMMENDATIONS:
================================================================================
"""
    
    # Add personalized recommendations
    if success_rate < 50:
        report += "- Focus on maintaining proper form rather than speed\n"
    if analytics['symmetry_violations'] > analytics['total_frames'] * 0.3:
        report += "- Work on keeping both arms moving symmetrically\n"
    if analytics['body_alignment_violations'] > analytics['total_frames'] * 0.3:
        report += "- Strengthen your core to maintain straight body alignment\n"
    if analytics['leg_position_violations'] > analytics['total_frames'] * 0.3:
        report += "- Keep your legs straight throughout the movement\n"
    if analytics['head_position_violations'] > analytics['total_frames'] * 0.3:
        report += "- Keep your head in neutral position, avoid looking up\n"
    
    if success_rate >= 80:
        report += "- Excellent form! Try increasing the number of pushups\n"
    elif success_rate >= 60:
        report += "- Good progress! Focus on consistency\n"
    
    # Save report to file
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    downloads_dir = os.path.expanduser('~/Downloads')
    os.makedirs(downloads_dir, exist_ok=True)
    report_filename = os.path.join(downloads_dir, f"pushup_report_{timestamp}.txt")
    
    report += f"""
================================================================================
📁 FILES GENERATED:
📄 Analytics Report: {report_filename}
📹 Analyzed Video: {output_video if output_video else 'Not saved'}

💡 TIP: Watch the analyzed video to see your form in real-time!
================================================================================
Report generated by Fitness Tracking App v2.0
================================================================================
"""
    with open(report_filename, 'w') as f:
        f.write(report)
    
    print(f"\n📊 ANALYTICS REPORT SAVED: {report_filename}")
    print(f"🏋️ SESSION COMPLETE: {analytics['valid_pushups']} valid pushups!")
    print(f"⭐ FORM SCORE: {avg_form_score:.1f}/4.0")
    
    return report_filename

    return angle


drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=4,color = (line_color))
drawing_spec_points = mp_drawing.DrawingSpec(thickness=5, circle_radius=4,color = (line_color))


detection_confidence = args.det
tracking_confidence = args.track

complexity = args.complexity

workout_type = args.workout_type


#For pushups
up_pos = None
down_pos = None
pushup_pos = None
display_pos = None

push_up_counter = 0

# Strict attempt tracking: an attempt is counted once per rep, only when the
# person starts from a held plank with arms extended and then clearly descends.
ATTEMPT_UP_ANGLE = 160      # arms extended
ATTEMPT_DESCENT_ANGLE = 120 # elbows must bend past this to count as an attempt
ATTEMPT_HOLD_FRAMES = 3     # consecutive frames needed to confirm each state
plank_up_frames = 0
descent_frames = 0
ready_for_attempt = False   # True after a confirmed plank-up start
attempt_counted = False     # True once this rep's attempt has been counted

# Analytics tracking
session_analytics = {
    'total_frames': 0,
    'good_form_frames': 0,
    'bad_form_frames': 0,
    'form_scores': [],
    'avg_arm_angles': [],
    'symmetry_violations': 0,
    'body_alignment_violations': 0,
    'leg_position_violations': 0,
    'head_position_violations': 0,
    'valid_pushups': 0,
    'attempted_pushups': 0,
    'session_start_time': time.time()
}





vid = cv2.VideoCapture(args.video)
#vid = cv2.VideoCapture(0)

width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid.get(cv2.CAP_PROP_FPS))

# Auto-generate output filename with timestamp if not specified
if args.output is None:
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    input_name = args.video.split('.')[0] if args.video else 'workout'
    os.makedirs("analysed", exist_ok=True)
    args.output = f"analysed/{input_name}_analyzed_{timestamp}.mp4"

print(f"📹 Saving analyzed video as: {args.output}")

# Use H264 codec for better compatibility
codec = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output, codec, fps, (width, height))

with mp_pose.Pose(
    min_detection_confidence=detection_confidence,
    min_tracking_confidence=tracking_confidence,
    model_complexity=complexity,smooth_landmarks = True ) as pose:

    while vid.isOpened():

        success, image = vid.read()

        #image = cv2.rotate(image,cv2.ROTATE_180)

        if not success:
            break
        #image = cv2.resize(image,(0,0),fx = 0.5, fy = 0.5)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
   
        image.flags.writeable = False
        results = pose.process(image)
        eyesVisible = False
        shoulderVisible = False


        
            #code for pose extraction
        try:
            
            landmarks = results.pose_landmarks.landmark

            


            #Check if both eyes are visible.

            left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]

            right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]

            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]

                #Get Tje Corridnates of Hip
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            #Put the Values for visibility 

            #visiblity for Eyes
            landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].visibility = 0

            #fOR NOSE
            landmarks[mp_pose.PoseLandmark.NOSE.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].visibility = 0



            #fOR eAR
            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].visibility = 0

            #print('LeftEye',left_visible)

            #Check if both shoulders are visible.
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
                #Midpointts
            midpoint_shoulder_x = (int(shoulder[0] * image_width )+ int(shoulder_r[0] * image_width))/2
            midpoint_shoulder_y = (int(shoulder[1] * image_height )+ int(shoulder_r[1] * image_height))/2

            midpoint_hip_x = (int(left_hip[0] * image_width )+ int(right_hip[0] * image_width))/2
            midpoint_hip_y = (int(left_hip[1] * image_height)+ int(right_hip[1] * image_height))/2


            based_mid_x = int((midpoint_shoulder_x + midpoint_hip_x)/2)
            based_mid_y = int((midpoint_shoulder_y + midpoint_hip_y)/2)

            neck_point_x = (int(nose[0] * image_width )+ int(midpoint_shoulder_x))/2
            neck_point_y = (int(nose[1] * image_height) + int(midpoint_shoulder_y))/2

            #print('Hip',left_hip)

            #angles 
            left_arm_angle = int(calculate_angle(shoulder, elbow, wrist))
            right_arm_angle = int(calculate_angle(shoulder_r, elbow_r, wrist_r))
            left_leg_angle = int(calculate_angle(left_hip, left_knee, left_ankle))

            right_leg_angle = int(calculate_angle(right_hip, right_knee, right_ankle))
            
            left_arm_length = np.linalg.norm(np.array(shoulder) - np.array(elbow))

            # ppm = 10.8

            # left_arm_motion = left_arm_angle* left_arm_length

            # left_arm_motion = left_arm_motion/ppm

            #newpoint_left = [left_hip[0] +5,right_hip[0] +5]

            mid_point_x = (int(left_hip[0] * image_width )+ int(right_hip[0] * image_width))/2
            mid_point_y = (int(left_hip[1] * image_height)+ int(right_hip[1] * image_height))/2
            #cv2.circle(image,(int(mid_point_x) ,int(mid_point_y +30 )),15,(0,255,255),-1)

            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility = 0


            if workout_type == 'PushUp':
                # ROBUST PUSHUP DETECTION - Multiple criteria for proper form
                
                # 1. ARM ANGLES - Both arms should be similar
                avg_arm_angle = (left_arm_angle + right_arm_angle) / 2
                arm_symmetry = abs(left_arm_angle - right_arm_angle) < 30  # Arms should be within 30° of each other
                
                # 2. BODY ALIGNMENT - Check if body is straight (plank position)
                # Calculate angle between shoulder-hip-knee line
                body_angle_left = int(calculate_angle([shoulder[0], shoulder[1]], 
                                                    [left_hip[0], left_hip[1]], 
                                                    [left_knee[0], left_knee[1]]))
                body_angle_right = int(calculate_angle([shoulder_r[0], shoulder_r[1]], 
                                                     [right_hip[0], right_hip[1]], 
                                                     [right_knee[0], right_knee[1]]))
                
                # Body should be relatively straight (140-180°)
                body_straight = (140 < body_angle_left < 180) and (140 < body_angle_right < 180)
                
                # 3. LEG POSITION - Legs should be relatively straight in pushup
                legs_straight = (left_leg_angle > 140) and (right_leg_angle > 140)
                
                # 4. HEAD POSITION - Nose should be forward/down, not looking up
                nose_hip_y_diff = abs(nose[1] - ((left_hip[1] + right_hip[1]) / 2))
                head_position_good = nose_hip_y_diff < 0.3  # Nose not too high above hips
                
                # 5. WRIST POSITION - Wrists should be roughly under shoulders in down position
                wrist_shoulder_distance_left = abs(wrist[0] - shoulder[0])
                wrist_shoulder_distance_right = abs(wrist_r[0] - shoulder_r[0])
                wrists_positioned = (wrist_shoulder_distance_left < 0.2) and (wrist_shoulder_distance_right < 0.2)
                
                # FORM SCORING SYSTEM
                form_score = 0
                form_checks = []
                
                if arm_symmetry:
                    form_score += 1
                    form_checks.append("Arms Symmetric")
                else:
                    form_checks.append("Arms Asymmetric")
                    session_analytics['symmetry_violations'] += 1
                    
                if body_straight:
                    form_score += 1
                    form_checks.append("Body Straight")
                else:
                    form_checks.append("Body Bent")
                    session_analytics['body_alignment_violations'] += 1
                    
                if legs_straight:
                    form_score += 1
                    form_checks.append("Legs Straight")
                else:
                    form_checks.append("Legs Bent")
                    session_analytics['leg_position_violations'] += 1
                    
                if head_position_good:
                    form_score += 1
                    form_checks.append("Head Position Good")
                else:
                    form_checks.append("Head Too High")
                    session_analytics['head_position_violations'] += 1
                
                # Track analytics
                session_analytics['total_frames'] += 1
                session_analytics['form_scores'].append(form_score)
                session_analytics['avg_arm_angles'].append(avg_arm_angle)
                
                if form_score >= 3:
                    session_analytics['good_form_frames'] += 1
                else:
                    session_analytics['bad_form_frames'] += 1
                
                # PUSHUP DETECTION with FORM REQUIREMENTS
                # Require at least 3/4 form criteria to be met
                good_form = form_score >= 3
                
                # STRICT ATTEMPT DETECTION
                in_plank = body_straight and legs_straight and arm_symmetry
                if avg_arm_angle > ATTEMPT_UP_ANGLE and in_plank:
                    plank_up_frames += 1
                    descent_frames = 0
                    if plank_up_frames >= ATTEMPT_HOLD_FRAMES:
                        ready_for_attempt = True
                        attempt_counted = False
                else:
                    plank_up_frames = 0
                    if ready_for_attempt and not attempt_counted and avg_arm_angle < ATTEMPT_DESCENT_ANGLE and in_plank:
                        descent_frames += 1
                        if descent_frames >= ATTEMPT_HOLD_FRAMES:
                            session_analytics['attempted_pushups'] += 1
                            attempt_counted = True
                            ready_for_attempt = False
                    else:
                        descent_frames = 0

                if avg_arm_angle > 160 and good_form:
                    up_pos = 'Up'
                    display_pos = f'Up (Form: {form_score}/4)'

                if avg_arm_angle < 90 and up_pos == 'Up' and good_form and attempt_counted:
                    down_pos = 'Down'
                    display_pos = f'Down (Form: {form_score}/4)'

                if avg_arm_angle > 160 and down_pos == 'Down' and good_form:
                    pushup_pos = "up"
                    display_pos = f"Valid Pushup! (Form: {form_score}/4)"
                    push_up_counter += 1
                    session_analytics['valid_pushups'] += 1
                    attempt_counted = False   # each attempt can produce at most one valid rep

                    up_pos = None
                    down_pos = None
                    pushup_pos = None
                    
                # If form is bad, reset positions to prevent cheating
                if not good_form:
                    if display_pos and 'Bad Form' not in str(display_pos):
                        display_pos = f'Bad Form ({form_score}/4)'

            cv2.line(image,(int(shoulder[0]* image_width),int(shoulder[1]* image_height)),(int(neck_point_x),int(neck_point_y)),(line_color),3)
            cv2.line(image,(int(shoulder_r[0]* image_width),int(shoulder_r[1]* image_height)),(int(neck_point_x),int(neck_point_y)),(line_color),3)
            cv2.line(image,(int(shoulder[0]* image_width),int(shoulder[1]* image_height)),(int(elbow[0]* image_width),int(elbow[1]* image_height)),(line_color),3)
            cv2.line(image,(int(shoulder_r[0]* image_width),int(shoulder_r[1]* image_height)),(int(elbow_r[0]* image_width),int(elbow_r[1]* image_height)),(line_color),3)

                #neck to mid point
            cv2.line(image,(int(neck_point_x),int(neck_point_y)),(int(based_mid_x),int(based_mid_y)),(line_color),3,cv2.LINE_4)

                #mid to hips
            cv2.line(image,(int(based_mid_x),int(based_mid_y)),(int(left_hip[0] * image_width ),(int(left_hip[1] * image_height))),(line_color),3,cv2.LINE_8)
            cv2.line(image,(int(based_mid_x),int(based_mid_y)),(int(right_hip[0] * image_width ),(int(right_hip[1] * image_height))),(line_color),3,cv2.LINE_8)


                ##neck point


            cv2.circle(image,(int(neck_point_x),int(neck_point_y)),4,(line_color),5)

                #create new circles at that place
            cv2.circle(image,(int(shoulder[0]* image_width),int(shoulder[1]* image_height)),4,(line_color),3)
            cv2.circle(image,(int(shoulder_r[0]* image_width),int(shoulder_r[1]* image_height)),4,(line_color),3)
                #mid point
            cv2.circle(image,(int(based_mid_x),int(based_mid_y)),4,(line_color),5)



            cv2.rectangle(image,(image_width,0),(image_width-350,280),(0,0,0),-1)
            cv2.putText(image,'Angles',(image_width-330,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
            cv2.putText(image, 'Left Elbow Angle: ' + str(left_arm_angle), (image_width-320, 70), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'Right Elbow Angle: ' + str(right_arm_angle), (image_width-320, 110), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'Left Knee Angle: ' + str(left_leg_angle), (image_width-320, 150), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'Right Knee Angle: ' + str(right_leg_angle), (image_width-320, 190), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'Push Up Counter: ' + str(push_up_counter), (image_width-340, 240), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, 'Position: ' + str(display_pos if display_pos else 'None'), (image_width-340, 270), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(image, 'Left arm motion: ' + str(left_arm_motion), (image_width-290, 230), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)







                




            


            


            
            


            #print('left eye',left_eye)

            #print('Is Eye Visible',eyesVisible)
            #print('Is Shoulder Visible',shoulderVisible)
        except:
            pass

        # Make image writable for drawing counter
        image.flags.writeable = True
        
        # CLEAN MINIMAL DISPLAY - Top bar only (doesn't block main video)
        # Semi-transparent overlay at the top
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image_width, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Essential info only - horizontal layout
        cv2.putText(image, f'PUSHUPS: {push_up_counter}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f'STATUS: {display_pos if display_pos else "Detecting..."}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Form score indicator - right side
        try:
            if 'form_score' in locals():
                score_color = (0,255,0) if form_score >= 3 else (0,165,255) if form_score >= 2 else (0,0,255)
                cv2.putText(image, f'FORM: {form_score}/4', (image_width-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2)
                cv2.putText(image, f'ANGLE: {int(avg_arm_angle)}°', (image_width-150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        except:
            cv2.putText(image, 'DETECTING POSE...', (image_width-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        
                    #writing angles
        
            #cv2.putText(image,"left elbow" + str(left_arm_angle),(int(image_width - 250),int(40)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
            #cv2.putText(image,str(right_arm_angle),(int(elbow_r[0]* image_width  -40),int(elbow_r[1]* image_height)),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,244,244),2,cv2.LINE_AA)


            #cv2.putText(image,str(left_leg_angle),(int(left_knee[0]* image_width + 40),int(left_knee[1]* image_height)),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),2,cv2.LINE_AA)
            #cv2.putText(image,str(right_leg_angle),(int(right_knee[0]* image_width - 40),int(right_knee[1]* image_height)),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),2,cv2.LINE_AA)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            drawing_spec_points,
            
        connection_drawing_spec=drawing_spec)

        fps =1.0/(time.time() - start_time)
        
        final_frame = image
        
        # Always save the analyzed video with analytics overlay
        out.write(final_frame)

        # Display resized version for real-time viewing
        final_frame_display = cv2.resize(final_frame,(0,0),fx = 0.5,fy = 0.5)

        cv2.imshow('Pose',final_frame_display)


        if cv2.waitKey(1) & 0xFF == 27:
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()

    # Generate analytics report at the end of session
    video_name = args.video if args.video else "webcam_session"
    report_file = generate_analytics_report(session_analytics, video_name, args.output)
    
    print(f"\n🎉 WORKOUT SESSION COMPLETED!")
    print(f"📄 Analytics report saved as: {report_file}")
    print(f"📹 Analyzed video saved as: {args.output}")
    print(f"📊 Review both files for comprehensive workout analysis!")

    # Automatically open the analyzed video after processing (Mac only)
    import subprocess
    if sys.platform == 'darwin' and sys.stdout.isatty():
        subprocess.run(['open', args.output])