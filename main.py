from mujoco import viewer
import mujoco
import numpy as np
import csv
import time
import pandas as pd

# length
"""
left_hip_pitch_joint
left_hip_roll_joint
left_hip_yaw_joint
left_knee_joint
left_ankle_pitch_joint
left_ankle_roll_joint
right_hip_pitch_joint
right_hip_roll_joint
right_hip_yaw_joint
right_knee_joint
right_ankle_pitch_joint
right_ankle_roll_joint
waist_yaw_joint
waist_roll_joint
waist_pitch_joint
left_shoulder_pitch_joint
left_shoulder_roll_joint
left_shoulder_yaw_joint
left_elbow_joint
left_wrist_roll_joint
left_wrist_pitch_joint
left_wrist_yaw_joint
left_hand_thumb_0_joint
left_hand_thumb_1_joint
left_hand_thumb_2_joint
left_hand_middle_0_joint
left_hand_middle_1_joint
left_hand_index_0_joint
left_hand_index_1_joint
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
right_wrist_roll_joint
right_wrist_pitch_joint
right_wrist_yaw_joint
right_hand_thumb_0_joint
right_hand_thumb_1_joint
right_hand_thumb_2_joint
right_hand_index_0_joint
right_hand_index_1_joint
right_hand_middle_0_joint
right_hand_middle_1_joint
"""

def load_trajectories_from_csv(file_path):
    df = pd.read_csv(file_path)
    
    trajectories = {}
    for column in df.columns:
        scale = 1.0
        if "0" in column:
            scale = 4.0

        if "thumb" in column:
            scale = -2.0

        
        trajectories[column] = (-df[column].values + 180.0)/90.0*scale
    
    return trajectories

trajectories = load_trajectories_from_csv('./policy/policy.csv')

def set_actuator_trajectory(model, name, values):
    id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    trajectories[id, :] = values

def inv_kinem(model, data, end_effector_id, goal, tol=1e-2, step_size=0.0, damping=0.15):
    error = np.subtract(goal, data.body(end_effector_id).xpos)
    if (np.linalg.norm(error) >= tol):
        #Calculate jacobian
        jacp = np.zeros((3, model.nv)) #translation jacobian
        jacr = np.zeros((3, model.nv)) #rotational jacobian

        end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_effector_id)
        mujoco.mj_jac(model, data, jacp, jacr, goal, end_effector_id)
        #Calculate delta of joint q
        n = jacp.shape[1]
        I = np.identity(n)
        product = jacp.T @ jacp + damping * I

        if np.isclose(np.linalg.det(product), 0):
            j_inv = np.linalg.pinv(product) @ jacp.T
        else:
            j_inv = np.linalg.inv(product) @ jacp.T

        delta_q = j_inv @ error

        #Compute next step
        q = data.qpos.copy()

        q += step_size * np.concatenate((delta_q, np.zeros(len(q)- model.nv)))
        
       
        #Check limits
        #check_joint_limits(data.qpos)
        
        return q[:model.nu]

def run(model_path):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    v = mujoco.viewer.launch_passive(model, data)
    #mujoco.viewer.launch(model)

    index = 0
    while True:
        t = int(index // 25) # int seconds

        #data.ctrl[:] = trajectory[:, t]

        ctrl = np.zeros(model.nu)
        for key, trajectory in trajectories.items():
           id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, key)
           ctrl[id] = trajectory[t]
        data.ctrl[:] = ctrl

        v.sync()
        index += 1
        mujoco.mj_step(model, data)

if __name__ == "__main__":
    model_path = './unitree_g1/scene_with_hands.xml'

    run(model_path)

