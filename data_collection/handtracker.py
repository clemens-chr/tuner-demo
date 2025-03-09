import cv2
import mediapipe as mp
import threading
import time
import numpy as np
import os
import csv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

csv_file = '/Users/ccc/dev/tuner/tuner-demo/policy/policy.csv'

def append_to_policy_csv(csv_file, dict_unitree):
    file_exists = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0

    with open(csv_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=dict_unitree.keys())

        if not file_exists:
            writer.writeheader()  # Write header only if file doesn't exist

        writer.writerow(dict_unitree)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

class MediaPipeTracker:
    def __init__(self, show=False, camera=0):
        self.is_recording = False
        self.cap = cv2.VideoCapture(camera)
        self.hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.keypoint_positions = None
        self.estimated_wrist_position = None  # Extra wrist point
        self.elbow_position = None  # Elbow position
        self.show = show
        self.running = False
        self.frame = None  # Shared frame for display in main thread
        self.thread = threading.Thread(target=self.run_tracker)

    def start_recording(self):
        self.is_recording = True
    
    def stop_recording(self):
        self.is_recording = False

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
        cv2.destroyAllWindows()

    def get_keypoint_positions(self):
        if self.estimated_wrist_position is not None and self.elbow_position is not None:
            z = self.keypoint_positions[0][2]
            self.estimated_wrist_position = np.array([self.estimated_wrist_position[0],self.estimated_wrist_position[1],z])
            return np.vstack((self.estimated_wrist_position, self.keypoint_positions))
        else:
            return None

    def run_tracker(self):
        while self.running and self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process hand and pose landmarks
            hand_results = self.hands.process(image)
            pose_results = self.pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            self.elbow_position = None  # Reset elbow position
            self.estimated_wrist_position = None  # Reset wrist position

            if pose_results.pose_landmarks:
                # Extract elbow position from pose
                landmarks = pose_results.pose_landmarks.landmark
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                self.elbow_position = np.array([right_elbow.x, right_elbow.y, right_elbow.z])

            
            if hand_results.multi_hand_landmarks:
                for hand_landmarks, hand_handedness in zip(
                    hand_results.multi_hand_landmarks, hand_results.multi_handedness
                ):
                    if hand_handedness.classification[0].label == "Left":  # Use "Right" hand labels are reversed
                        # Extract keypoint positions
                        self.keypoint_positions = np.array(
                            [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark],
                            dtype=np.float32,
                        )
                        

                        # Calculate the wrist and extrapolate
                        wrist = hand_landmarks.landmark[0]

                        if self.elbow_position is not None:
                            elbow_to_wrist = np.array([wrist.x, wrist.y, wrist.z]) - self.elbow_position
                            self.estimated_wrist_position = np.array([wrist.x, wrist.y, wrist.z]) - 0.4 * elbow_to_wrist

                        # Draw landmarks and the estimated wrist point
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                        )
                        

                        # Draw the estimated wrist position
                        if self.estimated_wrist_position is not None:
                            cv2.circle(
                                image,
                                (int(self.estimated_wrist_position[0] * image.shape[1]),
                                 int(self.estimated_wrist_position[1] * image.shape[0])),
                                5,
                                (255, 0, 0),  # Blue circle
                                -1,
                            )
                        

                        # Draw the elbow position
                        if self.elbow_position is not None:
                            cv2.circle(
                                image,
                                (int(self.elbow_position[0] * image.shape[1]),
                                 int(self.elbow_position[1] * image.shape[0])),
                                5,
                                (0, 255, 0),  # Green circle
                                -1,
                            )
                    
                        # show the angles for debug
                        mcp_joints = [5, 9, 13, 17]  # MCP joints
                        pip_joints = [6, 10, 14, 18]  # PIP joints
                        dip_joints = [7, 11, 15, 19]  # DIP joints
                        tip_joints = [8, 12, 16, 20]  # Fingertips
                        base = 0
                        
                        dict_angles = {
                            "index_mcp_angle": 0,
                            "middle_mcp_angle": 0,
                            "ring_mcp_angle": 0,
                            "pinky_mcp_angle": 0,
                            "index_pip_angle": 0,
                            "middle_pip_angle": 0,
                            "ring_pip_angle": 0,
                            "pinky_pip_angle": 0,
                            "thumb_mcp_angle": 0,
                            "thumb_pip_angle": 0,
                        }
                        
                        dict_unitree = {
                            "right_hand_thumb_0_joint": 0,
                            "right_hand_thumb_1_joint": 0,
                            "right_hand_thumb_2_joint": 0,
                            "right_hand_index_0_joint": 0,
                            "right_hand_index_1_joint": 0,
                            "right_hand_middle_0_joint": 0,
                            "right_hand_middle_1_joint": 0,
                        }
                        
                        mcp_angle_index = calculate_angle(
                            self.keypoint_positions[base],
                            self.keypoint_positions[mcp_joints[0]],
                            self.keypoint_positions[pip_joints[0]]
                        )
                        mcp_angle_middle = calculate_angle(
                            self.keypoint_positions[base],
                            self.keypoint_positions[mcp_joints[1]],
                            self.keypoint_positions[pip_joints[1]]
                        )
                        mcp_angle_ring = calculate_angle(
                            self.keypoint_positions[base],
                            self.keypoint_positions[mcp_joints[2]],
                            self.keypoint_positions[pip_joints[2]]
                        )
                        mcp_angle_pinky = calculate_angle(
                            self.keypoint_positions[base],
                            self.keypoint_positions[mcp_joints[3]],
                            self.keypoint_positions[pip_joints[3]]
                        )
                        
                        pip_angle_index = calculate_angle(
                            self.keypoint_positions[mcp_joints[0]],
                            self.keypoint_positions[pip_joints[0]],
                            self.keypoint_positions[tip_joints[0]]
                        )
                        
                        pip_angle_middle = calculate_angle(
                            self.keypoint_positions[mcp_joints[1]],
                            self.keypoint_positions[pip_joints[1]],
                            self.keypoint_positions[tip_joints[1]]
                        )
                        
                        pip_angle_ring = calculate_angle(
                            self.keypoint_positions[mcp_joints[2]],
                            self.keypoint_positions[pip_joints[2]],
                            self.keypoint_positions[tip_joints[2]]
                        )
                        
                        pip_angle_pinky = calculate_angle(
                            self.keypoint_positions[mcp_joints[3]],
                            self.keypoint_positions[pip_joints[3]],
                            self.keypoint_positions[tip_joints[3]]
                        )
                        
                        mcp_angle_thumb = calculate_angle(
                            self.keypoint_positions[base],
                            self.keypoint_positions[1],
                            self.keypoint_positions[2]
                        )
                        pip_angle_thumb = calculate_angle(
                            self.keypoint_positions[1],
                            self.keypoint_positions[2],
                            self.keypoint_positions[3]
                        )
                        dip_angle_thumb = calculate_angle(
                            self.keypoint_positions[2],
                            self.keypoint_positions[3],
                            self.keypoint_positions[4]
                        )
                        
                        dict_unitree["right_hand_thumb_0_joint"] = mcp_angle_thumb
                        dict_unitree["right_hand_thumb_1_joint"] = pip_angle_thumb
                        dict_unitree["right_hand_thumb_2_joint"] = dip_angle_thumb
                        
                        dict_unitree["right_hand_index_0_joint"] = (mcp_angle_index + mcp_angle_middle) / 2
                        dict_unitree["right_hand_index_1_joint"] = (pip_angle_index + pip_angle_middle) / 2
                        dict_unitree["right_hand_middle_0_joint"] = (mcp_angle_pinky + mcp_angle_ring) / 2
                        dict_unitree["right_hand_middle_1_joint"] = (pip_angle_pinky + pip_angle_ring) / 2
                        
                        if self.is_recording:
                            append_to_policy_csv(csv_file, dict_unitree)
                        
        
                        for i in range(4):
                            mcp_angle = 180 - calculate_angle(
                                self.keypoint_positions[base],
                                self.keypoint_positions[mcp_joints[i]],
                                self.keypoint_positions[pip_joints[i]]
                            )
                            
                            pip_angle = 180 - calculate_angle(
                                self.keypoint_positions[mcp_joints[i]],
                                self.keypoint_positions[pip_joints[i]],
                                self.keypoint_positions[tip_joints[i]]
                            )
                        
                            cv2.putText(image, f"MCP {i}: {int(mcp_angle)}", 
                                    (int(self.keypoint_positions[mcp_joints[i]][0] * image.shape[1]),
                                        int(self.keypoint_positions[mcp_joints[i]][1] * image.shape[0])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(image, f"PIP {i}: {int(pip_angle)}", 
                                    (int(self.keypoint_positions[pip_joints[i]][0] * image.shape[1]),
                                        int(self.keypoint_positions[pip_joints[i]][1] * image.shape[0])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


            # Update the frame for display
            self.frame = image if self.show else None
        self.cap.release()

def main():
    tracker = MediaPipeTracker(show=True, camera=0)
    
    tracker.start()
    try:
        while tracker.running:
            if tracker.frame is not None:
                if tracker.show:
                    cv2.imshow("MediaPipe Tracker", tracker.frame)
                # cv2.imshow("MediaPipe Tracker", tracker.frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    tracker.stop()
                    break

            keypoint_positions = tracker.get_keypoint_positions()
            if keypoint_positions is not None:
                print("Keypoints and Wrist Position:", keypoint_positions)
            else:
                if tracker.elbow_position is None:
                    print("Elbow position not found.")
                elif tracker.estimated_wrist_position is None:
                    print("Wrist position not found.")
            time.sleep(0.01)  # Small delay to avoid busy-waiting
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()