
import numpy as np
import mediapipe as mp
import math
import streamlit as st
import cv2
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle = np.abs(radians * 180.0 / np.pi)
    return angle

class CrunchStateMachine:
    def __init__(self):
        self.counter = 0
        self.is_crunching = False
        self.prev_left_angle = 0
        self.prev_right_angle = 0
        self.crunch_time = time.time()

    def update(self, angle_left, angle_right):
        if not self.is_crunching and time.time() - self.crunch_time > 1:
            if angle_left > 160 and angle_right > 160 and abs(angle_left - self.prev_left_angle) > 20 and abs(angle_right - self.prev_right_angle) > 20:
                self.is_crunching = True
                self.counter += 1
                self.crunch_time = time.time()
        else:
            if angle_left < 160 and angle_right < 160:
                self.is_crunching = False

        self.prev_left_angle = angle_left
        self.prev_right_angle = angle_right

def main():
    st.title("Crunch Counter using Streamlit")

    cap = cv2.VideoCapture(0)

    st.sidebar.title("Posture Analysis")
    posture_status = st.sidebar.empty()

    stframe = st.empty()
    crunch_state = CrunchStateMachine()

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        height, width, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            lmlist = results.pose_landmarks.landmark

            # Get landmarks for specific body parts
            left_shoulder = lmlist[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = lmlist[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = lmlist[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = lmlist[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_knee = lmlist[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = lmlist[mp_pose.PoseLandmark.RIGHT_KNEE.value]

            # Calculate angles for crunch detection
            angle_left = calculate_angle(left_shoulder, left_hip, left_knee)
            angle_right = calculate_angle(right_shoulder, right_hip, right_knee)

            # Update state machine
            crunch_state.update(angle_left, angle_right)

            # Draw the crunch count on the image
            cv2.putText(img, 'Count: ' + str(crunch_state.counter), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)

            # Draw landmarks and connections on the image
            mp_drawing = mp.solutions.drawing_utils
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Determine posture and display in sidebar
            posture_result = "Sitting" if (left_hip.y + right_hip.y) / 2 > (left_shoulder.y + right_shoulder.y) / 2 else "Standing"
            posture_status.text("Posture: " + posture_result)

        # Show the image in Streamlit
        stframe.image(img, channels="RGB", use_column_width=True)

    cap.release()

if __name__ == "__main__":
    main()
