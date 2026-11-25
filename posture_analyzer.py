import numpy as np
import cv2

class PostureAnalyzer:
    def __init__(self):
        self.mistakes_history = []
        self.suggestions_history = []

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def calculate_distance(self, a, b):
        """Calculate distance between two points"""
        return np.linalg.norm(np.array(a) - np.array(b))

    def analyze_pushup_posture(self, landmarks, image_width, image_height):
        """Analyze posture for pushups and return mistakes and suggestions"""
        mistakes = []
        suggestions = []

        # Extract key points
        left_shoulder = [landmarks[11].x * image_width, landmarks[11].y * image_height]
        right_shoulder = [landmarks[12].x * image_width, landmarks[12].y * image_height]
        left_elbow = [landmarks[13].x * image_width, landmarks[13].y * image_height]
        right_elbow = [landmarks[14].x * image_width, landmarks[14].y * image_height]
        left_wrist = [landmarks[15].x * image_width, landmarks[15].y * image_height]
        right_wrist = [landmarks[16].x * image_width, landmarks[16].y * image_height]
        left_hip = [landmarks[23].x * image_width, landmarks[23].y * image_height]
        right_hip = [landmarks[24].x * image_width, landmarks[24].y * image_height]
        left_knee = [landmarks[25].x * image_width, landmarks[25].y * image_height]
        right_knee = [landmarks[26].x * image_width, landmarks[26].y * image_height]
        nose = [landmarks[0].x * image_width, landmarks[0].y * image_height]

        # Calculate angles
        left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_leg_angle = self.calculate_angle(left_hip, left_knee, [left_knee[0], image_height])
        right_leg_angle = self.calculate_angle(right_hip, right_knee, [right_knee[0], image_height])

        # Body alignment check
        body_angle = self.calculate_angle(left_shoulder, [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2], left_knee)

        # Analyze mistakes with adjusted thresholds
        if abs(left_arm_angle - right_arm_angle) > 45:  # Increased from 30 to 45
            mistakes.append("Arm asymmetry")
            suggestions.append("Keep both arms moving symmetrically")

        if body_angle < 150 or body_angle > 180:  # Adjusted from 140-180 to 150-180
            mistakes.append("Body not straight")
            suggestions.append("Maintain a straight body line from head to heels")

        if left_leg_angle < 160 or right_leg_angle < 160:  # Increased from 140 to 160
            mistakes.append("Legs bent")
            suggestions.append("Keep legs straighter and engaged")

        if nose[1] < left_shoulder[1] - 30:  # Reduced threshold from 50 to 30
            mistakes.append("Head position incorrect")
            suggestions.append("Keep head in neutral position, avoid looking up")

        # Wrist position check with adjusted threshold
        wrist_shoulder_diff_left = abs(left_wrist[0] - left_shoulder[0])
        wrist_shoulder_diff_right = abs(right_wrist[0] - right_shoulder[0])
        if wrist_shoulder_diff_left > image_width * 0.2 or wrist_shoulder_diff_right > image_width * 0.2:  # Increased from 0.15 to 0.2
            mistakes.append("Wrists misaligned")
            suggestions.append("Position wrists directly under shoulders")

        return mistakes, suggestions

    def analyze_jumping_jack_posture(self, landmarks, image_width, image_height):
        """Analyze posture for jumping jacks"""
        mistakes = []
        suggestions = []

        # Extract key points
        left_shoulder = [landmarks[11].x * image_width, landmarks[11].y * image_height]
        right_shoulder = [landmarks[12].x * image_width, landmarks[12].y * image_height]
        left_wrist = [landmarks[15].x * image_width, landmarks[15].y * image_height]
        right_wrist = [landmarks[16].x * image_width, landmarks[16].y * image_height]
        left_hip = [landmarks[23].x * image_width, landmarks[23].y * image_height]
        right_hip = [landmarks[24].x * image_width, landmarks[24].y * image_height]
        left_ankle = [landmarks[27].x * image_width, landmarks[27].y * image_height]
        right_ankle = [landmarks[28].x * image_width, landmarks[28].y * image_height]

        # Check arm positions
        arms_up = left_wrist[1] < left_shoulder[1] and right_wrist[1] < right_shoulder[1]
        arms_down = left_wrist[1] > left_hip[1] and right_wrist[1] > right_hip[1]

        # Check leg positions
        legs_apart = self.calculate_distance(left_ankle, right_ankle) > self.calculate_distance(left_hip, right_hip) * 1.2
        legs_together = self.calculate_distance(left_ankle, right_ankle) < self.calculate_distance(left_hip, right_hip) * 1.4

        # Analyze mistakes
        if arms_up and not legs_apart:
            mistakes.append("Arms up but legs not apart")
            suggestions.append("When arms go up, jump feet apart simultaneously")

        if arms_down and not legs_together:
            mistakes.append("Arms down but legs not together")
            suggestions.append("When arms come down, jump feet together")

        # Check for lateral movement (common mistake)
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        if shoulder_width < image_width * 0.1:  # Shoulders too close
            mistakes.append("Shoulders not properly aligned")
            suggestions.append("Keep shoulders back and chest open")

        return mistakes, suggestions

    def analyze_vertical_jump_posture(self, landmarks, image_width, image_height):
        """Analyze posture for vertical jumps"""
        mistakes = []
        suggestions = []

        # Extract key points
        left_shoulder = [landmarks[11].x * image_width, landmarks[11].y * image_height]
        right_shoulder = [landmarks[12].x * image_width, landmarks[12].y * image_height]
        left_hip = [landmarks[23].x * image_width, landmarks[23].y * image_height]
        right_hip = [landmarks[24].x * image_width, landmarks[24].y * image_height]
        left_knee = [landmarks[25].x * image_width, landmarks[25].y * image_height]
        right_knee = [landmarks[26].x * image_width, landmarks[26].y * image_height]
        left_ankle = [landmarks[27].x * image_width, landmarks[27].y * image_height]
        right_ankle = [landmarks[28].x * image_width, landmarks[28].y * image_height]

        # Calculate angles
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)

        # Analyze mistakes
        if left_knee_angle > 150 or right_knee_angle > 150:
            mistakes.append("Insufficient knee bend")
            suggestions.append("Bend knees more deeply before jumping for better power")

        # Check landing posture
        torso_angle = self.calculate_angle(left_shoulder, [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2], left_knee)
        if torso_angle > 20:
            mistakes.append("Improper landing posture")
            suggestions.append("Land softly with knees bent and torso upright")

        # Check for arm swing
        # This would require tracking arm movement over time

        return mistakes, suggestions

    def get_realtime_suggestions(self, mistakes):
        """Convert mistakes to actionable suggestions for real-time display"""
        suggestion_map = {
            "Arm asymmetry": "Balance arms evenly",
            "Body not straight": "Straighten body alignment",
            "Legs bent": "Keep legs straight",
            "Head position incorrect": "Neutral head position",
            "Wrists misaligned": "Wrists under shoulders",
            "Arms up but legs not apart": "Jump feet apart with arms",
            "Arms down but legs not together": "Jump feet together",
            "Shoulders not properly aligned": "Open chest, shoulders back",
            "Insufficient knee bend": "Deeper knee bend",
            "Improper landing posture": "Soft landing with bent knees"
        }

        suggestions = [suggestion_map.get(mistake, mistake) for mistake in mistakes]
        return suggestions[:3]  # Limit to top 3 suggestions for display

    def get_ideal_pushup_posture(self, landmarks, image_width, image_height):
        """Generate ideal pushup posture landmarks based on current pose"""
        try:
            # For now, return the original landmarks since dual color visualization
            # is working without the ideal posture overlay
            return landmarks

        except Exception as e:
            print(f"Error generating ideal posture: {e}")
            return None

    def draw_suggestions_overlay(self, image, suggestions):
        """Draw suggestions on the image overlay"""
        if not suggestions:
            return image

        # Create semi-transparent overlay
        overlay = image.copy()
        h, w = image.shape[:2]

        # Background rectangle for suggestions
        cv2.rectangle(overlay, (10, h - 150), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        # Draw suggestions
        y_offset = h - 130
        cv2.putText(image, "Posture Suggestions:", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for i, suggestion in enumerate(suggestions[:3]):
            y_offset += 30
            # Draw suggestion text in white
            cv2.putText(image, f"• {suggestion}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            # Draw suggestion text in light green offset by 1 pixel for glow effect
            cv2.putText(image, f"• {suggestion}", (21, y_offset + 1),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (144, 238, 144), 1)

        return image
