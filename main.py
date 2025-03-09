from mujoco import viewer
import mujoco
import numpy as np
import csv
import time

def load_joint_angles(csv_file):
    joint_angles = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            joint_angles.append({
                'timestamp': float(row['timestamp']),
                'joint1': float(row['joint1']),
                'joint2': float(row['joint2']),
                'joint3': float(row['joint3'])
            })
    return joint_angles

def replay_joint_angles(model_path, joint_angles):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    

    start_time = time.time()
    for angle_set in joint_angles:
        current_time = time.time() - start_time
        while current_time < angle_set['timestamp']:
            current_time = time.time() - start_time

        data.ctrl[0] = angle_set['joint1']
        data.ctrl[1] = angle_set['joint2']
        data.ctrl[2] = angle_set['joint3']
        mujoco.mj_step(model, data)

if __name__ == "__main__":
    

    
    csv_file = '/Users/ccc/dev/tuner/tuner-demo/policy/policy.csv'
    model_path = '/Users/ccc/dev/tuner/tuner-demo/unitree_g1/scene_with_hands.xml'
    
    mujoco.viewer.launch(model_path)
    
    joint_angles = load_joint_angles(csv_file)
    replay_joint_angles(model_path, joint_angles)

