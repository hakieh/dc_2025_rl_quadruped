import numpy as np
import math
from utils.util import *

class UnitreeGo1Config():
    def __init__(self):
        self.fileName = "urdf/unitree_go1.urdf"

        # self.ground_contact_link = [
        #     "FL_lower_leg_2_upper_leg_joint",
        #     "FR_lower_leg_2_upper_leg_joint",
        #     "RL_lower_leg_2_upper_leg_joint",
        #     "RR_lower_leg_2_upper_leg_joint",
        # ]

        self.ground_contact_link = [
            "FL_lower_leg_2_foot_joint",
            "FR_lower_leg_2_foot_joint",
            "RL_lower_leg_2_foot_joint",
            "RR_lower_leg_2_foot_joint",

            # "FL_lower_leg_2_upper_leg_joint",
            # "FR_lower_leg_2_upper_leg_joint",
            # "RL_lower_leg_2_upper_leg_joint",
            # "RR_lower_leg_2_upper_leg_joint",
        ]

        #joint range from sdk hardware specification
        # lower_bound = [-1.047, -0.663, -2.721]
        # upper_bound = [1.047, 2.966, -0.837]
        #joint range from urdf
        # lower_bound = [-0.863, -0.686, -2.818]
        # upper_bound = [0.863, 4.501, -0.888]
        #use minimum range
        self.q_bound_default = dict([
            ("FR_hip_motor_2_chassis_joint", [-0.863, 0.863]),  # FR_HipX_joint
            ("FR_upper_leg_2_hip_motor_joint", [-0.663, 2.966]),# FR_HipY_joint
            ("FR_lower_leg_2_upper_leg_joint", [-2.721, -0.837]), # FR_Knee_joint
            ("FL_hip_motor_2_chassis_joint", [-0.863, 0.863]), # FL_HipX_joint
            ("FL_upper_leg_2_hip_motor_joint", [-0.663, 2.966]), # FL_HipY_joint
            ("FL_lower_leg_2_upper_leg_joint", [-2.721, -0.837]), # FL_Knee_joint
            ("RR_hip_motor_2_chassis_joint", [-0.863, 0.863]), # HR_HipX_joint
            ("RR_upper_leg_2_hip_motor_joint", [-0.663, 2.966]),# HR_HipY_joint
            ("RR_lower_leg_2_upper_leg_joint", [-2.721, -0.837]), # HR_Knee_joint
            ("RL_hip_motor_2_chassis_joint", [-0.863, 0.863]), # HL_HipX_joint
            ("RL_upper_leg_2_hip_motor_joint", [-0.663, 2.966]), # HL_HipY_joint
            ("RL_lower_leg_2_upper_leg_joint", [-2.721, -0.837]), # HL_Knee_joint
        ])

        self.q_nom_default = dict([
            ("FR_hip_motor_2_chassis_joint", 0),  # [-1.329, 1.181]
            ("FR_upper_leg_2_hip_motor_joint", -0.785),
            ("FR_lower_leg_2_upper_leg_joint", 1.57),  # [-0.23, 0.255]
            ("FL_hip_motor_2_chassis_joint", 0),
            ("FL_upper_leg_2_hip_motor_joint", -0.785),
            ("FL_lower_leg_2_upper_leg_joint", 1.57),
            ("RR_hip_motor_2_chassis_joint", 0),
            ("RR_upper_leg_2_hip_motor_joint", -0.785),
            ("RR_lower_leg_2_upper_leg_joint", 1.57),
            ("RL_hip_motor_2_chassis_joint", 0),
            ("RL_upper_leg_2_hip_motor_joint", -0.785),
            ("RL_lower_leg_2_upper_leg_joint", 1.57),
        ])

        self.q_dot_nom_default = dict([
            ("FR_hip_motor_2_chassis_joint", 0.0),  # [-1.329, 1.181]
            ("FR_upper_leg_2_hip_motor_joint", 0.0),
            ("FR_lower_leg_2_upper_leg_joint", 0.0),  # [-0.23, 0.255]
            ("FL_hip_motor_2_chassis_joint", 0.0),
            ("FL_upper_leg_2_hip_motor_joint", 0.0),
            ("FL_lower_leg_2_upper_leg_joint", 0.0),
            ("RR_hip_motor_2_chassis_joint", 0.0),
            ("RR_upper_leg_2_hip_motor_joint", 0.0),
            ("RR_lower_leg_2_upper_leg_joint", 0.0),
            ("RL_hip_motor_2_chassis_joint", 0.0),
            ("RL_upper_leg_2_hip_motor_joint", 0.0),
            ("RL_lower_leg_2_upper_leg_joint", 0.0),
        ])

        #max torque
        self.u_max_default = dict([
            ("FR_hip_motor_2_chassis_joint", 23.7), #108
            ("FR_upper_leg_2_hip_motor_joint", 23.7), #140
            ("FR_lower_leg_2_upper_leg_joint", 35.5),  #140
            ("FL_hip_motor_2_chassis_joint", 23.7),#108
            ("FL_upper_leg_2_hip_motor_joint", 23.7),#140
            ("FL_lower_leg_2_upper_leg_joint", 33.5),#140
            ("RR_hip_motor_2_chassis_joint", 23.7),#108
            ("RR_upper_leg_2_hip_motor_joint", 23.7),#140
            ("RR_lower_leg_2_upper_leg_joint", 33.5),#140
            ("RL_hip_motor_2_chassis_joint", 23.7),#108
            ("RL_upper_leg_2_hip_motor_joint", 23.7),#140
            ("RL_lower_leg_2_upper_leg_joint", 33.5),#140
        ])

        self.v_max_default = dict([
            ("FR_hip_motor_2_chassis_joint", 30.1),  # [-1.329, 1.181]
            ("FR_upper_leg_2_hip_motor_joint", 30.1),
            ("FR_lower_leg_2_upper_leg_joint", 20.06),  # [-0.23, 0.255]
            ("FL_hip_motor_2_chassis_joint", 30.1),
            ("FL_upper_leg_2_hip_motor_joint", 30.1),
            ("FL_lower_leg_2_upper_leg_joint", 20.06),
            ("RR_hip_motor_2_chassis_joint", 30.1),
            ("RR_upper_leg_2_hip_motor_joint", 30.1),
            ("RR_lower_leg_2_upper_leg_joint", 20.06),
            ("RL_hip_motor_2_chassis_joint", 30.1),
            ("RL_upper_leg_2_hip_motor_joint", 30.1),
            ("RL_lower_leg_2_upper_leg_joint", 20.06),
        ])

        #mass 13.100

        self.Kp_default = dict([
            ("FR_hip_motor_2_chassis_joint", 40),#100
            ("FR_upper_leg_2_hip_motor_joint", 40),
            ("FR_lower_leg_2_upper_leg_joint", 40),
            ("FL_hip_motor_2_chassis_joint", 40),
            ("FL_upper_leg_2_hip_motor_joint", 40),
            ("FL_lower_leg_2_upper_leg_joint", 40),
            ("RR_hip_motor_2_chassis_joint", 40),
            ("RR_upper_leg_2_hip_motor_joint", 40),
            ("RR_lower_leg_2_upper_leg_joint", 40),
            ("RL_hip_motor_2_chassis_joint", 40),
            ("RL_upper_leg_2_hip_motor_joint", 40),
            ("RL_lower_leg_2_upper_leg_joint", 40),
        ])
        self.Kd_default = dict([
            ("FR_hip_motor_2_chassis_joint", 3),  # [-1.329, 1.181]
            ("FR_upper_leg_2_hip_motor_joint", 3),
            ("FR_lower_leg_2_upper_leg_joint", 3),  # [-0.23, 0.255]
            ("FL_hip_motor_2_chassis_joint", 3),
            ("FL_upper_leg_2_hip_motor_joint", 3),
            ("FL_lower_leg_2_upper_leg_joint", 3),
            ("RR_hip_motor_2_chassis_joint", 3),
            ("RR_upper_leg_2_hip_motor_joint", 3),
            ("RR_lower_leg_2_upper_leg_joint", 3),
            ("RL_hip_motor_2_chassis_joint", 3),
            ("RL_upper_leg_2_hip_motor_joint", 3),
            ("RL_lower_leg_2_upper_leg_joint", 3),
        ])

        self.controlled_joints = [
            "FL_hip_motor_2_chassis_joint",
            "FL_upper_leg_2_hip_motor_joint",
            "FL_lower_leg_2_upper_leg_joint",
            "FR_hip_motor_2_chassis_joint",
            "FR_upper_leg_2_hip_motor_joint",
            "FR_lower_leg_2_upper_leg_joint",
            "RL_hip_motor_2_chassis_joint",
            "RL_upper_leg_2_hip_motor_joint",
            "RL_lower_leg_2_upper_leg_joint",
            "RR_hip_motor_2_chassis_joint",
            "RR_upper_leg_2_hip_motor_joint",
            "RR_lower_leg_2_upper_leg_joint",
            ]

        self.controllable_joints = [
            "FL_hip_motor_2_chassis_joint",
            "FL_upper_leg_2_hip_motor_joint",
            "FL_lower_leg_2_upper_leg_joint",
            "FR_hip_motor_2_chassis_joint",
            "FR_upper_leg_2_hip_motor_joint",
            "FR_lower_leg_2_upper_leg_joint",
            "RL_hip_motor_2_chassis_joint",
            "RL_upper_leg_2_hip_motor_joint",
            "RL_lower_leg_2_upper_leg_joint",
            "RR_hip_motor_2_chassis_joint",
            "RR_upper_leg_2_hip_motor_joint",
            "RR_lower_leg_2_upper_leg_joint",
        ]

        self.base_pos_nom_default = [0, 0, 0.35]
        self.base_orn_nom_default = [0, 0, 0, 1]
        self.base_euler_offset = []

        self.actionNumber = len(self.controlled_joints)
        self.action_bound = np.zeros((2,self.actionNumber))
        for i in range(self.actionNumber):
            joint_name = self.controlled_joints[i]
            self.action_bound[0][i] = self.q_bound_default[joint_name][0]  # lower bound
            self.action_bound[1][i] = self.q_bound_default[joint_name][1]  # upper bound


    #keypose initialization for fall recovery
        self.key_pose = []
        # nominal
        base_pos_nom = [0, 0, 0.35]
        base_orn_nom = euler_to_quat(0, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
        q_nom = dict([
            ("FR_hip_motor_2_chassis_joint", 0),  # [-1.329, 1.181]
            ("FR_upper_leg_2_hip_motor_joint", 0.785),
            ("FR_lower_leg_2_upper_leg_joint", -1.57),  # [-0.23, 0.255]
            ("FL_hip_motor_2_chassis_joint", 0),
            ("FL_upper_leg_2_hip_motor_joint", 0.785),
            ("FL_lower_leg_2_upper_leg_joint", -1.57),
            ("RR_hip_motor_2_chassis_joint", 0),
            ("RR_upper_leg_2_hip_motor_joint", 0.785),
            ("RR_lower_leg_2_upper_leg_joint", -1.57),
            ("RL_hip_motor_2_chassis_joint", 0),
            ("RL_upper_leg_2_hip_motor_joint", 0.785),
            ("RL_lower_leg_2_upper_leg_joint", -1.57),
        ])
        self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

        # stand tall
        base_pos_nom = [0, 0, 0.4]
        base_orn_nom = euler_to_quat(0, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
        q_nom = dict([
            ("FR_hip_motor_2_chassis_joint", 0),  # [-1.329, 1.181]
            ("FR_upper_leg_2_hip_motor_joint", 0.3),
            ("FR_lower_leg_2_upper_leg_joint", -0.66),  # [-0.23, 0.255]
            ("FL_hip_motor_2_chassis_joint", 0),
            ("FL_upper_leg_2_hip_motor_joint", 0.3),
            ("FL_lower_leg_2_upper_leg_joint", -0.66),
            ("RR_hip_motor_2_chassis_joint", 0),
            ("RR_upper_leg_2_hip_motor_joint", 0.3),
            ("RR_lower_leg_2_upper_leg_joint", -0.66),
            ("RL_hip_motor_2_chassis_joint", 0),
            ("RL_upper_leg_2_hip_motor_joint", 0.3),
            ("RL_lower_leg_2_upper_leg_joint", -0.66),
        ])
        self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

        # stand spread
        base_pos_nom = [0, 0, 0.3]
        base_orn_nom = euler_to_quat(0, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
        q_nom = dict([
            ("FR_hip_motor_2_chassis_joint", -0.39),  # [-1.329, 1.181]
            ("FR_upper_leg_2_hip_motor_joint", -0.18),
            ("FR_lower_leg_2_upper_leg_joint", -0.66),  # [-0.23, 0.255]
            ("FL_hip_motor_2_chassis_joint", 0.39),
            ("FL_upper_leg_2_hip_motor_joint", -0.18),
            ("FL_lower_leg_2_upper_leg_joint", -0.66),
            ("RR_hip_motor_2_chassis_joint", -0.39),
            ("RR_upper_leg_2_hip_motor_joint", 0.7),
            ("RR_lower_leg_2_upper_leg_joint", -0.66),
            ("RL_hip_motor_2_chassis_joint", 0.39),
            ("RL_upper_leg_2_hip_motor_joint", 0.7),
            ("RL_lower_leg_2_upper_leg_joint", -0.66),
        ])
        self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

        # crouch
        base_pos_nom = [0, 0, 0.13]
        base_orn_nom = euler_to_quat(0, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
        q_nom = dict([
            ("FR_hip_motor_2_chassis_joint", 0),  # [-1.329, 1.181]
            ("FR_upper_leg_2_hip_motor_joint", 1.66),
            ("FR_lower_leg_2_upper_leg_joint", -2.6),  # [-0.23, 0.255]
            ("FL_hip_motor_2_chassis_joint", 0),
            ("FL_upper_leg_2_hip_motor_joint", 1.66),
            ("FL_lower_leg_2_upper_leg_joint", -2.6),
            ("RR_hip_motor_2_chassis_joint", 0),
            ("RR_upper_leg_2_hip_motor_joint", 1.66),
            ("RR_lower_leg_2_upper_leg_joint", -2.6),
            ("RL_hip_motor_2_chassis_joint", 0),
            ("RL_upper_leg_2_hip_motor_joint", 1.66),
            ("RL_lower_leg_2_upper_leg_joint", -2.6),
        ])
        self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

        # knee support
        base_pos_nom = [0, 0, 0.23]
        base_orn_nom = euler_to_quat(0, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
        q_nom = dict([
            ("FR_hip_motor_2_chassis_joint", 0),  # [-1.329, 1.181]
            ("FR_upper_leg_2_hip_motor_joint", 0),
            ("FR_lower_leg_2_upper_leg_joint", -2.6),  # [-0.23, 0.255]
            ("FL_hip_motor_2_chassis_joint", 0),
            ("FL_upper_leg_2_hip_motor_joint", 0),
            ("FL_lower_leg_2_upper_leg_joint", -2.6),
            ("RR_hip_motor_2_chassis_joint", 0),
            ("RR_upper_leg_2_hip_motor_joint", 0),
            ("RR_lower_leg_2_upper_leg_joint", -2.6),
            ("RL_hip_motor_2_chassis_joint", 0),
            ("RL_upper_leg_2_hip_motor_joint", 0),
            ("RL_lower_leg_2_upper_leg_joint", -2.6),
        ])
        self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

        # trip lie, front leg stuck beneath
        base_pos_nom = [0, 0, 0.1]
        base_orn_nom = euler_to_quat(0, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
        q_nom = dict([
            ("FR_hip_motor_2_chassis_joint", 0),  # [-1.329, 1.181]
            ("FR_upper_leg_2_hip_motor_joint", 1.7),
            ("FR_lower_leg_2_upper_leg_joint", -0.66),  # [-0.23, 0.255]
            ("FL_hip_motor_2_chassis_joint", 0),
            ("FL_upper_leg_2_hip_motor_joint", 1.7),
            ("FL_lower_leg_2_upper_leg_joint", -0.66),
            ("RR_hip_motor_2_chassis_joint", 0),
            ("RR_upper_leg_2_hip_motor_joint", 1.884),
            ("RR_lower_leg_2_upper_leg_joint", -0.66),
            ("RL_hip_motor_2_chassis_joint", 0),
            ("RL_upper_leg_2_hip_motor_joint", 1.884),
            ("RL_lower_leg_2_upper_leg_joint", -0.66),
        ])
        self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])
        # Robot constantly flips on its back not need for initialization

        #back
        base_pos_nom = [0, 0, 0.08]
        base_orn_nom = euler_to_quat(3.14, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
        q_nom = dict([
            ("FR_hip_motor_2_chassis_joint", 0),  # [-1.329, 1.181]
            ("FR_upper_leg_2_hip_motor_joint", 0.74),
            ("FR_lower_leg_2_upper_leg_joint", -1.69),  # [-0.23, 0.255]
            ("FL_hip_motor_2_chassis_joint", 0),
            ("FL_upper_leg_2_hip_motor_joint", 0.74),
            ("FL_lower_leg_2_upper_leg_joint", -1.69),
            ("RR_hip_motor_2_chassis_joint", 0),
            ("RR_upper_leg_2_hip_motor_joint", 0.74),
            ("RR_lower_leg_2_upper_leg_joint", -1.69),
            ("RL_hip_motor_2_chassis_joint", 0),
            ("RL_upper_leg_2_hip_motor_joint", 0.74),
            ("RL_lower_leg_2_upper_leg_joint", -1.69),
        ])
        self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

        #right
        base_pos_nom = [0, 0, 0.16]
        base_orn_nom = euler_to_quat(1.57, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
        q_nom = dict([
            ("FR_hip_motor_2_chassis_joint", 0.39),  # [-1.329, 1.181]
            ("FR_upper_leg_2_hip_motor_joint", 0.6),
            ("FR_lower_leg_2_upper_leg_joint", -0.66),  # [-0.23, 0.255]
            ("FL_hip_motor_2_chassis_joint", -0.39),
            ("FL_upper_leg_2_hip_motor_joint", 0.3),
            ("FL_lower_leg_2_upper_leg_joint", -0.66),
            ("RR_hip_motor_2_chassis_joint", 0.39),  # [-1.329, 1.181]
            ("RR_upper_leg_2_hip_motor_joint", 0.6),
            ("RR_lower_leg_2_upper_leg_joint", -0.66),
            ("RL_hip_motor_2_chassis_joint", -0.39),
            ("RL_upper_leg_2_hip_motor_joint", 0.3),
            ("RL_lower_leg_2_upper_leg_joint", -0.66),
        ])
        self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

        #left
        base_pos_nom = [0, 0, 0.16]
        base_orn_nom = euler_to_quat(-1.57, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
        q_nom = dict([
            ("FR_hip_motor_2_chassis_joint", 0.39),  # [-1.329, 1.181]
            ("FR_upper_leg_2_hip_motor_joint", 0.3),
            ("FR_lower_leg_2_upper_leg_joint", -0.66),  # [-0.23, 0.255]
            ("FL_hip_motor_2_chassis_joint", -0.39),
            ("FL_upper_leg_2_hip_motor_joint", 0.6),
            ("FL_lower_leg_2_upper_leg_joint", -0.66),
            ("RR_hip_motor_2_chassis_joint", 0.39),  # [-1.329, 1.181]
            ("RR_upper_leg_2_hip_motor_joint", 0.3),
            ("RR_lower_leg_2_upper_leg_joint", -0.66),
            ("RL_hip_motor_2_chassis_joint", -0.39),
            ("RL_upper_leg_2_hip_motor_joint", 0.6),
            ("RL_lower_leg_2_upper_leg_joint", -0.66),
        ])
        self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])
