#!/usr/bin/python

import os
import inspect
import sys

import torch
# import copy

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
# os.sys.path.insert(0, currentdir)

import struct
import math
import numpy as np
import time
# import pybullet as p
# import pybullet
# import pybullet_data
# from pybullet_utils import bullet_client

# from motion_imitation.robots import a1_robot_velocity_estimator
from motion_imitation.robots import robot_config

from utils.filter_array import FilterClass, KalmanFilter, BinaryFilter
from utils.JointInterpolateArray import JointTrajectoryInterpolate
from utils.logger import *
from utils.util import *
from unitree_go1_standup_configuration import *

from go1_kinematic import *
from scipy.spatial.transform import Rotation as ROT


REAL_FLAG = True
print("currentdir",currentdir)
print(sys.path)
if REAL_FLAG == True:
    sys.path.append(currentdir+'/unitree_legged_sdk_official/lib/python/amd64')
    import robot_interface as sdk

print(sys.path)

#go1 configurations
DEFAULT_STAND_POSITION = [0,0.75,-1.57]*4#[0,0.67,-1.25]
DEFAULT_CROUCH_POSITION = [0,1.57,-2.5]*4
ABDUCTION_P_GAIN = 40#400
HIP_P_GAIN = 100#400
KNEE_P_GAIN = 40#00
ABDUCTION_D_GAIN = 3#10
HIP_D_GAIN = 3#10
KNEE_D_GAIN = 3#10
P_GAIN = 40 #TODO tune PD
D_GAIN = 3
Kp = np.array([100,100,100,   100,100,100,    100,100,100,    100,100,100])*40.0/100.0
Kd = np.array([3,3,3,    3,3,3,    3,3,3,    3,3,3])*1.0/3.0

# motor dict
# order of motor on hardware
d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
        'FL_0':3, 'FL_1':4, 'FL_2':5, 
        'RR_0':6, 'RR_1':7, 'RR_2':8, 
        'RL_0':9, 'RL_1':10, 'RL_2':11 }

# motor list
# order during command
motor_list = ['FL_0','FL_1','FL_2',
                'FR_0','FR_1','FR_2',
                'RL_0','RL_1','RL_2',
                'RR_0','RR_1','RR_2']

PosStopF = math.pow(10, 9)
VelStopF = 16000.0
HIGHLEVEL = 0xee
LOWLEVEL = 0xff




@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

class Go1_robot():

    def __del__(self):

        return

    def __init__(self,
                 pybullet_client = None,
                 control_freq=25.0,
                 motor_command_freq=500.0,
                 ):
        """Initializes the robot class."""
        # Initialize pd gain vector
        self.motor_kps = np.array([ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN] * 4)
        self.motor_kds = np.array([ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN] * 4)
        # if pybullet_client is None:
        #     self.pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        # else:
        #     self.pybullet_client=pybullet_client

        #environment related variables
        self.time_step = 1.0/motor_command_freq#time_step
        self._step_counter = 0

        self.r = -1
        self.control_freq = control_freq
        self.motor_command_freq = motor_command_freq
        self._control_loop_skip = int(motor_command_freq/control_freq)
        self._dt_motor_command = (1. / self.motor_command_freq)
        self._dt_control = (1./self.control_freq)
        self._dt_filter = self._dt_motor_command #filter time step
        self._dt_interpolate = self._dt_control#0.03 # time period for interpolation
        self.g = 9.81
        self.model_device = "cuda:0"
        self.dof_map_isaacsim = [ # from isaac sim simulation joint order to URDF order
                3, 0, 9, 6,
                4, 1, 10, 7,
                5, 2, 11, 8
            ] 
        
        self.dof_map_him = [ # from isaac sim simulation joint order to URDF order
                3,4,5,
                0,1,2,
                9,10,11,
                6,7,8
            ] 
        #self.default_pos = torch.tensor([0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5 ],device= self.model_device, dtype= torch.float32).unsqueeze(0)
        self.default_pos = torch.tensor([0.1, -0.1, 0.1, -0.1, 0.7, 0.7, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5 ],device= self.model_device, dtype= torch.float32).unsqueeze(0)
        self.actions = torch.zeros((1, 12), device= self.model_device, dtype= torch.float32)
        self.obs_buf_isaacsim = torch.zeros((1,270),device= self.model_device, dtype= torch.float32)
        self.obs_buf_isaacsim_new = torch.zeros((1,294),device= self.model_device, dtype= torch.float32)
        self.obs_buf_him = torch.zeros((1,270),device= self.model_device, dtype= torch.float32)
        self.joint_limits_low = torch.tensor([-1, -1, -1 ,-1, -2.5, -2.5, -2.5, -2.5, -2.0, -2.0, -2.0, -2.0],device="cpu")
        self.joint_limits_high =  torch.tensor([1, 1, 1 ,1, 3.5, 3.5, 3.5, 3.5, 1, 1, 1, 1],device="cpu")
        self.joint_limits_low_him = torch.tensor([-1, -2.5, -2.0 ]*4,device="cpu")
        self.joint_limits_high_him =  torch.tensor([1,3.5, 1 ]*4,device="cpu")
        # Robot state variables
        self._init_complete = False
        self._base_orientation = None
        self._raw_state = None
        self._last_raw_state = None
        self._base_angular_velocity = np.zeros(3)
        self._base_angular_velocity_fusion = np.zeros(3) #fuse gyro with angular velocity calculated from leg
        self._base_angular_acceleration = np.zeros(3)
        # reordered to fit neural network
        self._motor_angles = np.zeros(12)
        self._motor_velocities = np.zeros(12)
        self._motor_torques = np.zeros(12)
        self._motor_accelerations = np.zeros(12)
        self._foot_force = np.zeros(4)
        self._foot_force_est = np.zeros(4)
        # raw order from sensor readings
        self._raw_motor_angles = np.zeros(12)
        self._raw_motor_velocities = np.zeros(12)
        self._raw_motor_accelerations = np.zeros(12)
        self._raw_motor_torques = np.zeros(12)
        self._raw_foot_force = np.zeros(4)
        self._raw_foot_force_est = np.zeros(4)
        self._joint_states = None
        self._last_reset_time = time.time()
        # self._velocity_estimator = a1_robot_velocity_estimator.VelocityEstimator(robot=self,accelerometer_variance=0.1,
        #        sensor_variance=0.1,
        #        initial_variance=0.1,
        #        moving_window_filter_size=120)
        self.wireless_remote = None
        self.header = None
        self.button = np.zeros(16)
        self.lx = 0
        self.rx = 0
        self.ry = 0
        self.L2 = 0
        self.ly = 0

        self.imu_0ffset = [0,0,0] #offset gravity from imu
        self.foot_force_offset = [0,0,0,0]

        if REAL_FLAG == True:
            # Initiate UDP for robot state and actions
            self.udp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
            self.safe = sdk.Safety(sdk.LeggedType.Go1)
            self.cmd = sdk.LowCmd()
            self.state = sdk.LowState()
            self.udp.InitCmdData(self.cmd)
            self._motor_control_mode = 1

        # RL variables
        self.target_vel = np.zeros(6)
        self.filtered_target_vel = np.zeros(6)
        self._stateDim = 9+2+12*2+3
        self._obsDim = 45
        self._actionDim = 12
        self.observation = np.zeros(self._stateDim)
        self.observation_filtered = np.zeros(self._stateDim)
        self.policy_action = np.array(DEFAULT_STAND_POSITION)#policy output
        self.command_action = np.zeros(self._actionDim)#filtered action

        self.power_limit = 1
        self.position_limit = 0.1

        #Robot basic configuration
        self.robotConfig = UnitreeGo1Config()
        self.q_bound_default = self.robotConfig.q_bound_default
        self.Kp = np.array(Kp)#self.robotConfig.Kp_default
        self.Kd = np.array(Kd)#self.robotConfig.Kd_default
        self.controlled_joints = self.robotConfig.controlled_joints

        # nominal joint configuration
        self.u_max = self.robotConfig.u_max_default
        self.v_max = self.robotConfig.v_max_default
        self.q_nom = self.robotConfig.q_nom_default
        self.nominal_motor_angles = np.array([0, 0.8, -1.65] * 4)#np.array([0, 0.9, -1.8] * 4)
        self.crouch_motor_angles = np.array([0, 1.4, -2.66] * 4)  # np.array([0, 0.9, -1.8] * 4)
        self.him_default_angle = torch.tensor(self.nominal_motor_angles,device="cuda:0")

        #foot force estimation
        bin = 11#bin range 0N~120N, 12N per bin
        self.foot_force_prior = np.ones((4,bin))/bin

        self._setupFilter()

        self._init_complete = True

        #control related
        self.FSM_state = 0
        return

    def _setupFilter(self):

        #TODO tune parameter to reduce oscillation
        # solution1 state cutoff 8Hz, action cutoff 6Hz

        #filter using array
        self.state_filter_method = FilterClass(self._stateDim)
        self.state_filter_method.butterworth(self._dt_motor_command, 10, 1)  # sample period, cutoff frequency, order

        filter_order = 1
        self.action_filter_method = FilterClass(self._actionDim)
        # self.action_filter_method.butterworth(1./self.motor_command_freq, 4, filter_order)  # sample period, cutoff frequency, order
        self.action_filter_method.butterworth(self._dt_motor_command, 8,
                                              filter_order)  # sample period, cutoff frequency, order

        self.target_vel_filter_method = FilterClass(6)
        # self.target_vel_filter_method.butterworth(self._dt_motor_command, 0.25, 1)  # sample period, cutoff frequency, order
        self.target_vel_filter_method.butterworth(self._dt_motor_command, 1,
                                                  1)  # sample period, cutoff frequency, order

        # self._velocity_estimator.reset()

    def reset(self, reload_urdf=True, default_motor_angles=None, reset_time=5.0):

        # self._velocity_estimator.reset()
        self._state_action_counter = 0
        self._step_counter = 0
        self._last_reset_time = time.time()

        #wait and receive data
        wait_time = 1  # process filter for 1 sec
        time_step = 1.0 / self.motor_command_freq
        for t in np.arange(0, wait_time, time_step):
            self.receiveState()
            time.sleep(time_step)
            self.udp.send()#send to recive

        #init filter
        wait_time = 1 #process filter for 1 sec
        time_step = 1.0 / self.motor_command_freq
        self.policy_action = default_motor_angles #initialize action filter
        for t in np.arange(0, wait_time, time_step):
            self.receiveObservation()
            self.getObservation()
            self.filterObservation()
            self.filterAction()
            time.sleep(time_step)
            self.udp.send()


        return self.observation_filtered

    def terminate(self):
        self._is_alive = False
        return

    def convertWirelessRemote(self):
        self.wireless_remote = self._raw_state.wirelessRemote

        def uint8ToBit(byte_list):  # list of 2 bytes to 8 bit
            bytes = np.squeeze(np.array(byte_list, dtype=np.uint8))
            b = np.unpackbits(bytes, axis=0, bitorder='little')
            # b = np.concatenate(b)
            # print(b)
            return np.squeeze(b)

        self.button = uint8ToBit(self.wireless_remote[2:4])

        def uint8ToFloat32(byte_list):  # list of 4 bytes to float
            bytes = np.squeeze(np.array(byte_list, dtype='<u1'))
            bytes = bytes.tobytes('C')  # C order
            aa = bytearray(bytes)
            floats = struct.unpack('<f', aa)
            # print(floats)
            return np.squeeze(floats)

        self.lx = uint8ToFloat32(self.wireless_remote[4:8])
        self.rx = uint8ToFloat32(self.wireless_remote[8:12])
        self.ry = uint8ToFloat32(self.wireless_remote[12:16])
        self.L2 = uint8ToFloat32(self.wireless_remote[16:20])
        self.ly = uint8ToFloat32(self.wireless_remote[20:24])
        return

    def receiveObservation(self):
        """Receives observation from robot.
        Synchronous ReceiveObservation is not supported in A1,
        so changging it to noop instead.
        """
        if REAL_FLAG == True:
            self.receiveState()

        else:
            # self.state.motorState =
            self.state.imu.quaternion = [0,0,0,1]
            self.state.imu.gyroscope = [0,0,0]
            self.state.imu.accelerometer = [0,0,0]
            self.state.imu.rpy = [0,0,0]
            self.state.footForce = [0,0,0,0]

        self.convertWirelessRemote()
        self.updateTargetVel()
        self.filterTargetVel()

        # Convert quaternion from wxyz to xyzw, which is default for Pybullet.
        q = self.state.imu.quaternion
        self._base_orientation = np.array([q[1], q[2], q[3], q[0]])
        motor_angle_temp = np.zeros(12)
        self._raw_motor_angles = np.array([motor.q for motor in self.state.motorState[:12]])
        # print( self._raw_motor_angles)
        motor_angle_temp[0:4] = [self._raw_motor_angles[3],self._raw_motor_angles[0],self._raw_motor_angles[9],self._raw_motor_angles[6]]
        motor_angle_temp[4:8] = [self._raw_motor_angles[4],self._raw_motor_angles[1],self._raw_motor_angles[10],self._raw_motor_angles[7]]
        motor_angle_temp[8:12] = [self._raw_motor_angles[5],self._raw_motor_angles[2],self._raw_motor_angles[11],self._raw_motor_angles[8]]
        # motor_angle_temp[9:12] = self._raw_motor_angles[6:9]
        # print("raw joint-process joint 0",
        #       np.array([motor.q for motor in self.state.motorState[:12]]) - self._motor_angles)
        self._motor_angles = torch.tensor(motor_angle_temp, device=self.model_device)

        motor_angle_temp[0:3] = self._raw_motor_angles[3:6]
        motor_angle_temp[3:6] = self._raw_motor_angles[0:3]
        motor_angle_temp[6:9] = self._raw_motor_angles[9:12]
        motor_angle_temp[9:12] = self._raw_motor_angles[6:9]
        # print("raw joint-process joint 0",
        #       np.array([motor.q for motor in self.state.motorState[:12]]) - self._motor_angles)
        self._motor_angles_stand = motor_angle_temp
        # print("raw joint-process joint 1", np.array([motor.q for motor in self.state.motorState[:12]]) - self._motor_angles)
        motor_velocity_temp = np.zeros(12)
        self._raw_motor_velocities = np.array([motor.dq for motor in self.state.motorState[:12]])
        motor_velocity_temp[0:4] = [self._raw_motor_velocities[3],self._raw_motor_velocities[0],self._raw_motor_velocities[9],self._raw_motor_velocities[6]]
        motor_velocity_temp[4:8] = [self._raw_motor_velocities[4],self._raw_motor_velocities[1],self._raw_motor_velocities[10],self._raw_motor_velocities[7]]
        motor_velocity_temp[8:12] = [self._raw_motor_velocities[5],self._raw_motor_velocities[2],self._raw_motor_velocities[11],self._raw_motor_velocities[8]]
        # motor_velocity_temp[9:12] = self._raw_motor_velocities[6:9]
        self._motor_velocities = motor_velocity_temp
        motor_torque_temp = np.zeros(12)
        self._raw_motor_torques = np.array([motor.tauEst for motor in self.state.motorState[:12]])
        motor_torque_temp[0:3] = self._raw_motor_torques[3:6]
        motor_torque_temp[3:6] = self._raw_motor_torques[0:3]
        motor_torque_temp[6:9] = self._raw_motor_torques[9:12]
        motor_torque_temp[9:12] = self._raw_motor_torques[6:9]
        self._motor_torques = motor_torque_temp
        foot_force_temp = np.zeros(4)
        self._raw_foot_force = np.array(self._raw_state.footForce)
        foot_force_temp[0] = self._raw_foot_force[1]
        foot_force_temp[1] = self._raw_foot_force[0]
        foot_force_temp[2] = self._raw_foot_force[3]
        foot_force_temp[3] = self._raw_foot_force[2]
        self._foot_force = foot_force_temp
        # self._joint_states = np.array(
        #     list(zip(self._motor_angles, self._motor_velocities)))
        self._base_angular_acceleration = np.array(self._raw_state.imu.gyroscope).copy()-self._base_angular_velocity
        self._base_angular_velocity = np.array(self._raw_state.imu.gyroscope).copy()
        self._base_euler = np.array(self._raw_state.imu.rpy).copy()

        #fusion
        blend = (np.sum(self._foot_force)-300)/600 #
        blend = np.clip(blend,0,1)*0.5
        self._base_angular_velocity_fusion = np.array(self._raw_state.imu.gyroscope).copy()
        # self._base_angular_velocity_fusion[2] = (1.0-blend)*self._base_angular_velocity[2]+\
        #                                      blend*self._velocity_estimator.yaw_vel_from_foot

        # if self._init_complete:
        #     # self._SetRobotStateInSim(self._motor_angles, self._motor_velocities)
        #     self._velocity_estimator.update(self._raw_state)

        return

    def receiveState(self):
        self.udp.Recv()
        self.udp.GetRecv(self.state)
        self._raw_state = self.state
        # print("receive data ...")
        # self._raw_state = copy.deepcopy(self.state)

    # def getObservation(self):
    #     x_observation = np.zeros((self._stateDim,)) #create a new numpy array instance
    #     """
    #     Extract desired observation data from received observation

    #     """
    #     base_quat = self._base_orientation #pybullet order xyzw
    #     # base_euler = self.pybullet_client.getEulerFromQuaternion(base_quat)
    #     # base_euler_from_quat = np.squeeze(base_euler)
    #     # print("base_euler:,",base_euler)
    #     # print("base_euler_from_quat")
    #     # base_euler = np.array([0,0,0])
    #     if (base_quat == np.array([0,0,0,0])).all():
    #         base_quat = np.array([0,0,0,1])
    #     rot = ROT.from_quat(base_quat)
    #     # rot1 = rot.as_euler("XYZ",degrees=False)
    #     rot2 = rot.as_euler("ZYX",degrees=False)
    #     # print("rot2:", rot2)
    #     rot_res = np.array([0,0,0])
    #     rot_res[0] = rot2[2]
    #     rot_res[1] = rot2[1]
    #     rot_res[2] = rot2[0]
    #     rot = rot_res
    #     # print("rot:", rot)
        
    #     base_orn = np.squeeze(rot)
    #     # base_orn = np.squeeze(base_euler)
    #     # self.base_euler_from_quat = np.squeeze(base_euler)
    #     self.base_euler_from_quat = np.squeeze(rot)

    #     Rz = rotZ(base_orn[2])
    #     Rz_i = np.linalg.inv(Rz)
    #     self.Rz_i = Rz_i
    #     R = quat_to_rot(base_quat)
    #     R_i = np.linalg.inv(R)

    #     # base_pos_vel = np.squeeze(self._velocity_estimator.estimated_velocity)

    #     # base linear velocity
    #     base_pos_vel = np.array(base_pos_vel)
    #     base_pos_vel.resize(1, 3)
    #     # base_pos_vel_base = np.transpose(R_i @ base_pos_vel.transpose())  # base velocity in base (pelvis) frame
    #     base_pos_vel_yaw = np.transpose(Rz_i @ base_pos_vel.transpose())  # base velocity in adjusted yaw frame
    #     self.base_pos_vel_yaw = base_pos_vel_yaw

    #     x_observation[0] = base_pos_vel_yaw[0][0]  # pelvis_x_dot
    #     x_observation[1] = base_pos_vel_yaw[0][1]  # pelvis_y_dot
    #     x_observation[2] = base_pos_vel_yaw[0][2]  # pelvis_z_dot

    #     gravity = np.array([0,0,-1])
    #     # gravity_quat = self.pybullet_client.getQuaternionFromEuler([0,0,0])
    #     gravity_quat = np.array([0,0,0,1])
    #     # invBasePos, invBaseQuat = self.pybullet_client.invertTransform([0,0,0], base_quat)
    #     # invBasePos = np.array([0,0,0])
    #     # invBaseQuat = np.array([0,0,0,0])
    #     orn_4 = ROT.from_quat(base_quat)
    #     inverse_orn = orn_4.inv()

    #     invBasePos = orn_4.apply([0,0,0])
    #     invBaseQuat = inverse_orn.as_quat()
    #     # print("====================================")
    #     # print("inverse_pos", inverse_pos)
    #     # print("inverse_orn:", inverse_orn)
    #     # print("invBasePos",invBasePos)
    #     # print("invBaseQuat",invBaseQuat)
    #     # print("====================================")
    #     #gravity vector in base frame
    #     # gravityPosInBase, gravityQuatInBase = self.pybullet_client.multiplyTransforms(invBasePos, invBaseQuat, gravity, gravity_quat)

    #     orn1 = ROT.from_quat(invBaseQuat)
    #     orn2 = ROT.from_quat(gravity_quat)

    #     res_orn = orn1 * orn2
    #     gravityPosInBase = invBasePos + orn1.apply(gravity)
    #     gravityQuatInBase = res_orn.as_quat()
    #     # gravityPosInBase_test = res_pos
    #     # print("res_pos:", res_pos)
    #     # print("res_orn:", res_orn)
    #     # print("gravityPosInBase",gravityPosInBase)
    #     # print("gravityQuatInBase",gravityQuatInBase)
    #     # print("====================================")
    #     # gravityPosInBase = np.array([0,0,0])
    #     # gravityQuatInBase = np.array([0,0,0,0])
    #     self.gravityPosInBase = np.squeeze(gravityPosInBase)
    #     x_observation[3] = gravityPosInBase[0]
    #     x_observation[4] = gravityPosInBase[1]
    #     x_observation[5] = gravityPosInBase[2]

    #     #base angular velocity
    #     # base_orn_vel = np.array(self._base_angular_velocity)
    #     # base_orn_vel.resize(1,3)
    #     # base_orn_vel_base = np.transpose(R_i @ base_orn_vel.transpose())
    #     # self.base_orn_vel = base_orn_vel
    #     # self.base_orn_vel_base = base_orn_vel_base

    #     base_orn_vel_base = np.array(self._base_angular_velocity)
    #     # base_orn_vel_base = np.array(self._base_angular_velocity_fusion)
    #     base_orn_vel_base.resize(1,3)

    #     self.base_orn_vel_base = base_orn_vel_base

    #     vel_factor = 1
    #     delay_dt = 0.0#0.003#0.004ms
    #     x_observation[6] = base_orn_vel_base[0][0]  # pelvis_roll_dot
    #     x_observation[7] = base_orn_vel_base[0][1]  # pelvis_pitch_dot
    #     x_observation[8] = base_orn_vel_base[0][2]  # pelvis_yaw_dot
    #     #TODO tune yaw velocity

    #     x_observation[9] = 0#phase_sin
    #     x_observation[10] = 0#phase_cos

    #     x_observation[11] = self.filtered_target_vel[0]#x
    #     x_observation[12] = self.filtered_target_vel[1]#y
    #     x_observation[13] = self.filtered_target_vel[5]#*np.clip(1-self.gravityPosInBase[2],0,1)#yaw

    #     index = 11+3#15

    #     #unitree motor command order
    #     for i in range(self._actionDim):
    #         name = self.controlled_joints[i]

    #         x_observation[index] = self._motor_velocities[i] / self.v_max[name]  # velocity normalized by max vel
    #         x_observation[index+self._actionDim] = self._motor_angles[i] # position
    #         # x_observation[index+self._actionDim*2] = self.prev_action[i] #past action
    #         index+=1 #counter

    #     x_observation = np.nan_to_num(np.squeeze(x_observation))
    #     self.observation = np.array(x_observation)

    #     return np.array(x_observation)

    def getObservation_isaacsim(self):
        x_observation = np.zeros((self._obsDim,)) #create a new numpy array instance
        """
        Extract desired observation data from received observation

        """
        # 
        base_quat = self._base_orientation #pybullet order xyzw
        if (base_quat == np.array([0,0,0,0])).all():
            base_quat = np.array([0,0,0,1])
        base_quat = torch.tensor(base_quat,device=self.model_device,dtype=torch.float32).unsqueeze(0)
        gravity = torch.tensor([0,0,-1],device=self.model_device,dtype=torch.float32).unsqueeze(0)
        projected_gravity = quat_rotate_inverse(base_quat,gravity).to(self.model_device)


        base_orn_vel_base = np.array(self._base_angular_velocity)
        base_orn_vel_base.resize(1,3)
        self.base_orn_vel_base = torch.tensor(base_orn_vel_base,device=self.model_device,dtype=torch.float32)

        dof_pos = torch.tensor([self._raw_motor_angles[self.dof_map_isaacsim[i]] for i in range(12)],device=self.model_device,dtype=torch.float32).unsqueeze(0)
        dof_vel = torch.tensor([self._raw_motor_velocities[self.dof_map_isaacsim[i]] for i in range(12)],device=self.model_device,dtype=torch.float32).unsqueeze(0)
        #print(dof_vel,"==")
        
        command = torch.tensor([1.00, 0, 0],device=self.model_device,dtype=torch.float32).unsqueeze(0)

        
        obs_step = torch.cat(  (self.base_orn_vel_base*0.25,
                                    projected_gravity,
                                    command,
                                    (dof_pos - self.default_pos),
                                    dof_vel*0.05,
                                    self.actions),
                                    dim=-1
                                   )
        current_obs = torch.tensor(obs_step,device=self.model_device,dtype=torch.float32)
        self.obs_buf_isaacsim = torch.cat((self.obs_buf_isaacsim[:, 45:],current_obs[:, :] ), dim=-1)
        return self.obs_buf_isaacsim
    def getObservation_isaacsim_new(self):
        x_observation = np.zeros((self._obsDim,)) #create a new numpy array instance
        """
        Extract desired observation data from received observation

        """
        # 
        base_quat = self._base_orientation #pybullet order xyzw
        if (base_quat == np.array([0,0,0,0])).all():
            base_quat = np.array([0,0,0,1])
        base_quat = torch.tensor(base_quat,device=self.model_device,dtype=torch.float32).unsqueeze(0)
        gravity = torch.tensor([0,0,-1],device=self.model_device,dtype=torch.float32).unsqueeze(0)
        projected_gravity = quat_rotate_inverse(base_quat,gravity).to(self.model_device)


        base_orn_vel_base = np.array(self._base_angular_velocity)
        base_orn_vel_base.resize(1,3)
        self.base_orn_vel_base = torch.tensor(base_orn_vel_base,device=self.model_device,dtype=torch.float32)

        dof_pos = torch.tensor([self._raw_motor_angles[self.dof_map_isaacsim[i]] for i in range(12)],device=self.model_device,dtype=torch.float32).unsqueeze(0)
        #print("dof_pos:",dof_pos)
        #print("default:",self.default_pos)
        dof_vel = torch.tensor([self._raw_motor_velocities[self.dof_map_isaacsim[i]] for i in range(12)],device=self.model_device,dtype=torch.float32).unsqueeze(0)
        #print(dof_vel,"==")
        command = torch.tensor([1.00, 0, 0],device=self.model_device,dtype=torch.float32).unsqueeze(0)
        contact_real=self.GetFootContacts()
        contact = torch.tensor([contact_real[1],contact_real[0],contact_real[3],contact_real[2]],device=self.model_device).unsqueeze(0)
        
        obs_step = torch.cat(  (self.base_orn_vel_base*0.25,
                                    projected_gravity,
                                    command,
                                    (dof_pos - self.default_pos),
                                    dof_vel*0.05,
                                    self.actions,
                                    contact),
                                    dim=-1
                                   )
        current_obs = torch.tensor(obs_step,device=self.model_device,dtype=torch.float32)
        self.obs_buf_isaacsim_new = torch.cat((self.obs_buf_isaacsim_new[:, 49:],current_obs[:, :] ), dim=-1)
        return self.obs_buf_isaacsim_new

    
    def getObservation_him(self):
        x_observation = np.zeros((self._obsDim,)) #create a new numpy array instance
        """
        Extract desired observation data from received observation

        """
        # 
        base_quat = self._base_orientation #pybullet order xyzw
        if (base_quat == np.array([0,0,0,0])).all():
            base_quat = np.array([0,0,0,1])
        base_quat = torch.tensor(base_quat,device=self.model_device,dtype=torch.float32).unsqueeze(0)
        gravity = torch.tensor([0,0,-1],device=self.model_device,dtype=torch.float32).unsqueeze(0)
        projected_gravity = quat_rotate_inverse(base_quat,gravity).to(self.model_device)


        base_orn_vel_base = np.array(self._base_angular_velocity)
        base_orn_vel_base.resize(1,3)
        self.base_orn_vel_base = torch.tensor(base_orn_vel_base,device=self.model_device,dtype=torch.float32)

        dof_pos = torch.tensor([self._raw_motor_angles[self.dof_map_him[i]] for i in range(12)],device=self.model_device,dtype=torch.float32).unsqueeze(0)
        dof_vel = torch.tensor([self._raw_motor_velocities[self.dof_map_him[i]] for i in range(12)],device=self.model_device,dtype=torch.float32).unsqueeze(0)
        # print(dof_pos,"==")
        command = torch.tensor([0.5, 0, 0],device=self.model_device,dtype=torch.float32).unsqueeze(0)

        obs_step = torch.cat(  (   command,
                                    self.base_orn_vel_base*0.25,
                                    projected_gravity,
                                    (dof_pos - self.him_default_angle),
                                    dof_vel*0.05,
                                    self.actions),
                                    dim=-1
                                   )
        current_obs = torch.tensor(obs_step,device=self.model_device,dtype=torch.float32)
        self.obs_buf_him = torch.cat((current_obs[:, :],self.obs_buf_isaacsim[:, :-45] ), dim=-1)
        return self.obs_buf_him

    


    def filterObservation(self):
        observation = self.getObservation()

        #filter using array
        self.state_filter_method.applyFilter(observation)

        observation_filtered = self.state_filter_method.y[0]
        #replace filtered state with unfiltered state for user provided data
        observation_filtered[9:11] = [0,0]#self.unfiltered_phase
        observation_filtered[0:3] = observation[0:3]  # no filter for velocity
        # observation_filtered[self.stateNumber-self.actionNumber:self.stateNumber] = self.unfiltered_prev_action
        self.observation_filtered = np.array(observation_filtered)
        return observation_filtered

    def setSafetyLimit(self, power_limit, position_limit=None):
        self.power_limit = power_limit
        # if position_limit != None:
        self.position_limit = position_limit #limit popsition difference
        return

    def limitAction(self, action=None, limit=None):#23.7/40.0
        if action is not None:
            self.policy_action = action
        if limit is not None:
            self.position_limit = limit
        delta = self.policy_action*0.25+self.default_pos-self._motor_angles

        delta = torch.clip(delta, -self.position_limit, self.position_limit)
        self.policy_action = (self._motor_angles+delta-self.default_pos)*4
        return self.policy_action

    def limitAction_stand(self, action=None, limit=None):#23.7/40.0
        if action is not None:
            self.policy_action = action
        if limit is not None:
            self.position_limit = limit
        delta = self.policy_action-torch.tensor(self._motor_angles_stand,device="cuda:0")

        delta = torch.clip(delta, -self.position_limit, self.position_limit)
        self.policy_action = torch.tensor(self._motor_angles_stand,device="cuda:0")+delta
        return self.policy_action


    def filterAction(self, action=None):
        if action is not None:
            self.policy_action = action
        filtered_action = self.action_filter_method.applyFilter(self.policy_action)
        self.command_action = np.squeeze(filtered_action) #filtered action used for command
        return filtered_action

    def step(self, motor_commands=None, motor_control_mode=None, user_target = [0,0,0,0,0,0]):
        self.target_vel = np.squeeze(user_target) #update target user command
        action = motor_commands #update motor commands
        repeat = int(self.motor_command_freq/self.control_freq)
        step_time = 1.0/self.motor_command_freq
        #apply filter
        for i in range(repeat):
            start_time = time.time()
            #apply action filter

            #apply control action
            self.applyAction()

            #kalman filter

            #state filter

            end_time = time.time()
            process_time = end_time-start_time #process duration

            pause_time = max(step_time-process_time,0)
            time.sleep(pause_time)

        return

    def crouch(self, reset_time=3):
        self.udp.Send()
        self.receiveObservation()
        self.udp.Send()
        # current_motor_angles = self.GetMotorAngles()
        current_motor_angles = self._motor_angles#use angles with flipped left and right leg
        # crouch in 3 seconds, and keep the behavior in this way.
        reset_time = min(reset_time, 3)
        for t in np.arange(0, reset_time, self.time_step):
            blend_ratio = min(t / reset_time, 1)
            action = blend_ratio * self.crouch_motor_angles + (
                    1 - blend_ratio) * current_motor_angles
            self.applyAction(action, robot_config.MotorControlMode.POSITION)
            time.sleep(self.time_step)

        return


    def stand(self, reset_time = 3): #nominal standing posture
        self.udp.Send()
        self.receiveObservation()
        self.udp.Send()
        # current_motor_angles = self._motor_angles_stand
        
        current_motor_angles = self._motor_angles_stand#use angles with flipped left and right leg
        print("current:",current_motor_angles)
        # Stand up in 3 seconds, and keep the behavior in this way.
        reset_time_3 = min(reset_time, 2)
        for t in np.arange(0, reset_time, self.time_step):
            tic = time.perf_counter()
            blend_ratio = min(t / reset_time_3, 1)
            action = blend_ratio * self.nominal_motor_angles + (
                    1 - blend_ratio) * current_motor_angles
            current_angles = self._motor_angles_stand
            # print("current:",current_angles)
            # print("action:",action)
            self.applyActionStand(action)
            toc = time.perf_counter()
            duration = toc-tic
            delay = np.clip(self.time_step-duration,0.0,self.time_step)
            print("delay:",delay)
            time.sleep(delay)
        return
    def keep_stand(self):
        # if actions.any():
        # self.applyActionStand(actions)
        # else:
        self.applyActionStand(self.nominal_motor_angles)

    def keep_stand2(self,shake_time=20):
        # if actions.any():
        # self.applyActionStand(actions)
        # else:
        for t in np.arange(0, shake_time, self.time_step):
            print(t)
            angle = np.sin(t*np.pi)*30/180*np.pi
            print(angle)
            print("policy angle:",self.nominal_motor_angles+angle)
            self.receiveObservation()
            print("current angle:",self._motor_angles_stand)
            # self.applyActionStand(self.nominal_motor_angles+angle)
            self.applyAction(angle)
            time.sleep(self.time_step)
    def rest(self, reset_time=10.0):
        self.udp.Send()
        self.receiveObservation()
        self.udp.Send()
        self.Kp = [30,30,30,  30,30,30,  30,30,30,  30,30,30]#small P
        #damping
        for t in np.arange(0, reset_time, self.time_step):
            self.receiveObservation()
            # current_motor_angles = self.GetMotorAngles()
            current_motor_angles = self._motor_angles_stand  # use angles with flipped left and right leg
            action = current_motor_angles            
            self.applyActionStand(action)
            time.sleep(self.time_step)

        return

    def crouchAction(self,t, reset_time):
        blend_ratio = min(t / reset_time, 1)
        current_motor_angles = self._motor_angles  # use angles with flipped left and right leg
        action = blend_ratio * self.crouch_motor_angles + (
                1 - blend_ratio) * current_motor_angles
        self.command_action = np.squeeze(action)
        return

    def standAction(self,t, reset_time):
        blend_ratio = min(t / reset_time, 1)
        current_motor_angles = self._motor_angles  # use angles with flipped left and right leg
        action = blend_ratio * self.nominal_motor_angles + (
                1 - blend_ratio) * current_motor_angles
        self.command_action = np.squeeze(action)
        return

    def restAction(self,t=None, reset_time=10):
        current_motor_angles = self._motor_angles  # use angles with flipped left and right leg
        action = current_motor_angles
        self.command_action = np.squeeze(action)
        return

    def Terminate(self):
        self._is_alive = False
        self.safe.PowerProtect(self.cmd, self.state, 1)
        return

    def stateMachineAction(self):
        self.receiveObservation()
        self.getObservation()
        self.filterObservation()

        if self.FSM_state ==0: #go to crouching state
            self._step_counter = self._step_counter+1.0
            self.crouchAction(self._step_counter/self.motor_command_freq,3)
            if self._step_counter>5*self.motor_command_freq:
                self.FSM_state = 1 # transit to new state reset
                self._step_counter = 0.0
        elif self.FSM_state == 1: # go to standing state
            self._step_counter = self._step_counter+1.0
            self.standAction(self._step_counter/self.motor_command_freq,3)
            if self._step_counter>5*self.motor_command_freq:
                self.FSM_state = 3 # transit to new state reset
                self._step_counter = 0.0
        elif self.FSM_state == 2: # go to policy state
            self._step_counter = self._step_counter+1
            self.filterAction(self.policy_action)
        elif self.FSM_state == 3: # resting state
            self._step_counter = self._step_counter + 1
            self.restAction(0,3)
        else:
            self._step_counter = self._step_counter + 1
            self.restAction(0,3)
        return

    def applyAction(self, actions, power_limit=None):
        """Clips and then apply the motor commands using the motor model.
        Args:
          motor_commands: np.array. Can be motor angles, torques, hybrid commands,
            or motor pwms (for Minitaur only).
          motor_control_mode: A MotorControlMode enum.
        """
        # self.actions = actions
        actions[0]*=0.5
        actions[1]*=0.5
        actions[2]*=0.5
        actions[3]*=0.5

        temp  = torch.tensor([self._raw_motor_angles[self.dof_map_isaacsim[i]] for i in range(12)],device=self.model_device,dtype=torch.float32).unsqueeze(0)
        # print("current angle:",temp-self.default_pos)
        action_scale = 0.25# 0.25
        # print("policy:",actions)
        actions = actions * action_scale + self.default_pos
        # print
        robot_coordinates_action = torch.clip(
            actions.cpu(),
            self.joint_limits_low,
            self.joint_limits_high
        )
        # print()
        # command_temp = torch.zeros(12,device=self.model_device,dtype=torch.float32)
        # #flip command for left and right leg
        # command_temp = 
        # print("command_temp", command_temp)
        robot_coordinates_action = np.squeeze(np.array(robot_coordinates_action))
        # print(robot_coordinates_action)
        # motor_commands = command_temp

        # if motor_control_mode is None:
        #   motor_control_mode = self._motor_control_mode
  
        for sim_joint_idx in range(12):
            real_joint_idx = self.dof_map_isaacsim[sim_joint_idx]
            # motor_name = motor_list[motor_id]
            # self.cmd.motorCmd[d[motor_name]].q = motor_commands[motor_id]
            # self.cmd.motorCmd[d[motor_name]].dq = 0
            # self.cmd.motorCmd[d[motor_name]].Kp = P_GAIN#Kp[motor_id]
            # self.cmd.motorCmd[d[motor_name]].Kd = D_GAIN#Kd[motor_id]
            # self.cmd.motorCmd[d[motor_name]].tau = 0

            #self.cmd.motorCmd[real_joint_idx].mode=10
            self.cmd.motorCmd[real_joint_idx].q = robot_coordinates_action[sim_joint_idx]
            self.cmd.motorCmd[real_joint_idx].dq = 0
            self.cmd.motorCmd[real_joint_idx].Kp = 30 #self.Kp[sim_joint_idx]#P_GAIN#Kp[motor_id]
            self.cmd.motorCmd[real_joint_idx].Kd = 1 #self.Kd[sim_joint_idx]#D_GAIN#Kd[motor_id]
            self.cmd.motorCmd[real_joint_idx].tau = 0

        self.cmd.motorCmd[0].Kd=self.cmd.motorCmd[3].Kd=self.cmd.motorCmd[6].Kd=self.cmd.motorCmd[9].Kd=3
        #self.cmd.motorCmd[0].Kp=self.cmd.motorCmd[3].Kp=self.cmd.motorCmd[6].Kp=self.cmd.motorCmd[9].Kp=30

        if power_limit is not None:
            self.power_limit = power_limit
        #safety
        if self.power_limit<=10:#apply power protect only when value is between(0,10)
            self.safe.PowerProtect(self.cmd, self.state, self.power_limit)#1
        self.safe.PositionLimit(self.cmd)
        # self.safe.PositionProtect(self.cmd, self.state, 1)

        self.udp.SetSend(self.cmd)
        self.udp.Send()

        return

    def applyAction_new(self, actions, power_limit=None):
        """Clips and then apply the motor commands using the motor model.
        Args:
          motor_commands: np.array. Can be motor angles, torques, hybrid commands,
            or motor pwms (for Minitaur only).
          motor_control_mode: A MotorControlMode enum.
        """
        # self.actions = actions
        # actions[0]*=0.5
        # actions[1]*=0.5
        # actions[2]*=0.5
        # actions[3]*=0.5

        temp  = torch.tensor([self._raw_motor_angles[self.dof_map_isaacsim[i]] for i in range(12)],device=self.model_device,dtype=torch.float32).unsqueeze(0)
        # print("current angle:",temp-self.default_pos)
        action_scale = 0.25# 0.25
        # print("policy:",actions)
        # actions = actions * action_scale + self.default_pos
        # print
        robot_coordinates_action = torch.clip(
            actions.cpu(),
            self.joint_limits_low,
            self.joint_limits_high,
        )
        # print()
        # command_temp = torch.zeros(12,device=self.model_device,dtype=torch.float32)
        # #flip command for left and right leg
        # command_temp = 
        # print("command_temp", command_temp)
        robot_coordinates_action = np.squeeze(np.array(robot_coordinates_action))
        # print(robot_coordinates_action)
        # motor_commands = command_temp

        # if motor_control_mode is None:
        #   motor_control_mode = self._motor_control_mode

        
        for sim_joint_idx in range(12):
            real_joint_idx = self.dof_map_isaacsim[sim_joint_idx]
            # motor_name = motor_list[motor_id]
            # self.cmd.motorCmd[d[motor_name]].q = motor_commands[motor_id]
            # self.cmd.motorCmd[d[motor_name]].dq = 0
            # self.cmd.motorCmd[d[motor_name]].Kp = P_GAIN#Kp[motor_id]
            # self.cmd.motorCmd[d[motor_name]].Kd = D_GAIN#Kd[motor_id]
            # self.cmd.motorCmd[d[motor_name]].tau = 0

            self.cmd.motorCmd[real_joint_idx].q = robot_coordinates_action[sim_joint_idx]
            self.cmd.motorCmd[real_joint_idx].dq = 0
            self.cmd.motorCmd[real_joint_idx].Kp = 25 #self.Kp[sim_joint_idx]#P_GAIN#Kp[motor_id]
            self.cmd.motorCmd[real_joint_idx].Kd = 0.5 #self.Kd[sim_joint_idx]#D_GAIN#Kd[motor_id]
            self.cmd.motorCmd[real_joint_idx].tau = 0

        self.cmd.motorCmd[0].Kd=self.cmd.motorCmd[3].Kd=self.cmd.motorCmd[6].Kd=self.cmd.motorCmd[9].Kd=3
        #self.cmd.motorCmd[0].Kp=self.cmd.motorCmd[3].Kp=self.cmd.motorCmd[6].Kp=self.cmd.motorCmd[9].Kp=30

        if power_limit is not None:
            self.power_limit = power_limit
        #safety
        if self.power_limit<=10:#apply power protect only when value is between(0,10)
            self.safe.PowerProtect(self.cmd, self.state, self.power_limit)#1
        self.safe.PositionLimit(self.cmd)
        # self.safe.PositionProtect(self.cmd, self.state, 1)

        self.udp.SetSend(self.cmd)
        self.udp.Send()

        return


    def applyAction_him(self, actions, power_limit=None):

        """Clips and then apply the motor commands using the motor model.
        Args:
          motor_commands: np.array. Can be motor angles, torques, hybrid commands,
            or motor pwms (for Minitaur only).
          motor_control_mode: A MotorControlMode enum.
        """
        self.actions = actions
        
        
        temp  = torch.tensor([self._raw_motor_angles[self.dof_map_him[i]] for i in range(12)],device=self.model_device,dtype=torch.float32).unsqueeze(0)
        # print("current angle:",temp-self.default_pos)
        action_scale = 0.25# 0.25
        # print("policy:",actions)
        actions = actions * action_scale + self.him_default_angle
        # print
        robot_coordinates_action = torch.clip(
            actions.cpu(),
            self.joint_limits_low_him,
            self.joint_limits_high_him,
        )
        # command_temp = torch.zeros(12,device=self.model_device,dtype=torch.float32)
        # #flip command for left and right leg
        # command_temp = 
        # print("command_temp", command_temp)
        robot_coordinates_action = np.squeeze(np.array(robot_coordinates_action))
        # print(robot_coordinates_action)
        # motor_commands = command_temp

        # if motor_control_mode is None:
        #   motor_control_mode = self._motor_control_mode

  
        for sim_joint_idx in range(12):
            real_joint_idx = self.dof_map_him[sim_joint_idx]
            # motor_name = motor_list[motor_id]
            # self.cmd.motorCmd[d[motor_name]].q = motor_commands[motor_id]
            # self.cmd.motorCmd[d[motor_name]].dq = 0
            # self.cmd.motorCmd[d[motor_name]].Kp = P_GAIN#Kp[motor_id]
            # self.cmd.motorCmd[d[motor_name]].Kd = D_GAIN#Kd[motor_id]
            # self.cmd.motorCmd[d[motor_name]].tau = 0

            self.cmd.motorCmd[real_joint_idx].q = robot_coordinates_action[sim_joint_idx]
            self.cmd.motorCmd[real_joint_idx].dq = 0
            self.cmd.motorCmd[real_joint_idx].Kp = 30 #self.Kp[sim_joint_idx]#P_GAIN#Kp[motor_id]
            self.cmd.motorCmd[real_joint_idx].Kd = 1.5 #self.Kd[sim_joint_idx]#D_GAIN#Kd[motor_id]
            self.cmd.motorCmd[real_joint_idx].tau = 0


        if power_limit is not None:
            self.power_limit = power_limit
        #safety
        if self.power_limit<=10:#apply power protect only when value is between(0,10)
            self.safe.PowerProtect(self.cmd, self.state, self.power_limit)#1
        self.safe.PositionLimit(self.cmd)
        # self.safe.PositionProtect(self.cmd, self.state, 1)
        self.udp.SetSend(self.cmd)
        self.udp.Send()
        return


    def calPDTorque(self,action=None, torque_limit_scale = 1):#calculate PD torque
        if action is not None:
            self.command_action = action
        torque = P_GAIN*np.squeeze(self.command_action-self._motor_angles)-D_GAIN*self._motor_velocities
        torque_limit = np.array([23.7,23.7,35.5]*4)*torque_limit_scale
        torque = np.clip(torque, -torque_limit,torque_limit)

        return torque

    def updateTargetVel(self,tar_x=0,tar_y=0,tar_yaw=0):
        # self.target_vel[0] = tar_x
        # self.target_vel[1] = tar_y
        # self.target_vel[5] = tar_yaw

        self.target_vel[0] = np.clip(self.ly,-1,2)#np.clip(self.ly*4,0,4)
        self.target_vel[1] = np.clip(-self.lx,-1,1)*0.5
        self.target_vel[5] = np.clip(-self.rx * np.pi,-np.pi,np.pi)*0.5
        return

    def filterTargetVel(self, target_vel=None):
        if target_vel is not None:
            self.target_vel = target_vel
        filtered_target_vel = self.target_vel_filter_method.applyFilter(self.target_vel)
        self.filtered_target_vel = np.squeeze(filtered_target_vel)
        return filtered_target_vel

    def GetMotorAngles(self): #motor angle in raw order
        # temp = np.array([motor.q for motor in self.state.motorState[:12]])
        # return temp.copy()
        return self._raw_motor_angles.copy()

    def GetMotorVelocities(self): #motor angle in raw order
        # temp = np.array([motor.dq for motor in self.state.motorState[:12]])
        # return temp.copy()
        return self._raw_motor_velocities.copy()

    def applyActionStand(self, motor_commands, motor_control_mode=None, power_limit=None):

        """Clips and then apply the motor commands using the motor model.
        Args:
          motor_commands: np.array. Can be motor angles, torques, hybrid commands,
            or motor pwms (for Minitaur only).
          motor_control_mode: A MotorControlMode enum.
        """
        command_temp = np.zeros(12)
        #flip command for left and right leg
        command_temp[0:3] = motor_commands[3:6]
        command_temp[3:6] = motor_commands[0:3]
        command_temp[6:9] = motor_commands[9:12]
        command_temp[9:12] = motor_commands[6:9] #adjust command order
        # print("command_temp", command_temp)

        motor_commands = command_temp

        if motor_control_mode is None:
          motor_control_mode = self._motor_control_mode

        
        for motor_id in range(12):
            # motor_name = motor_list[motor_id]
            # self.cmd.motorCmd[d[motor_name]].q = motor_commands[motor_id]
            # self.cmd.motorCmd[d[motor_name]].dq = 0
            # self.cmd.motorCmd[d[motor_name]].Kp = P_GAIN#Kp[motor_id]
            # self.cmd.motorCmd[d[motor_name]].Kd = D_GAIN#Kd[motor_id]
            # self.cmd.motorCmd[d[motor_name]].tau = 0

            self.cmd.motorCmd[motor_id].q = motor_commands[motor_id]
            self.cmd.motorCmd[motor_id].dq = 0
            self.cmd.motorCmd[motor_id].Kp = 30 #self.Kp[motor_id]#P_GAIN#Kp[motor_id]
            self.cmd.motorCmd[motor_id].Kd =  1 #self.Kd[motor_id]#D_GAIN#Kd[motor_id]
            self.cmd.motorCmd[motor_id].tau = 0
        if power_limit is not None:
            self.power_limit = power_limit
        #safety
        if self.power_limit<=10:#apply power protect only when value is between(0,10)
            self.safe.PowerProtect(self.cmd, self.state, self.power_limit)#1
        self.safe.PositionLimit(self.cmd)
        # self.safe.PositionProtect(self.cmd, self.state, 1)

        self.udp.SetSend(self.cmd)
        self.udp.Send()

        return
    

    def GetBaseRollPitchYaw(self):
        return self._base_euler.copy()

    def GetBaseRollPitchYawRate(self):
        return np.array(self._raw_state.imu.gyroscope).copy()

    def GetBaseVelocity(self):
        return self._velocity_estimator.estimated_velocity.copy()

    def GetFootContacts(self): #foor contact in raw order
        #return np.array(self._raw_state.footForce) > 200#20
        # use processed foot force, left and right leg are flipped
        # return np.array(self._foot_force) > 300  # 20
        # return np.array(self._raw_state.footForce) > 150#20 #for flat ground
        # return np.array(self._raw_state.footForce) > 250 #for grassy terrain
        return np.array(self._raw_state.footForce)-self.foot_force_offset>50 #offset sensor drift

    def GetFootForceEst(self): # get foot force estimation from torque
        #0kg 7.42
        #12kg 266.689
        #12+1.25 293.482
        # 12+2.5 320.124
        # 12+3.75 350.271
        # 12+5.0 380.616
        self._foot_force*9.81/88.8638

        return

    def GetFootForceEstFromTorque(self): # get foot force estimation from torque
        #0kg 1.1232
        #12kg 24.239
        #12+1.25 25.724
        # 12+2.5 27.99
        # 12+3.75  30.1557
        # 12+5.0  32.585

        knee_torque = np.array([self._motor_torques[2],self._motor_torques[5],self._motor_torques[8],self._motor_torques[11]])
        knee_pos = np.array([self._motor_angles[2],self._motor_angles[5],self._motor_angles[8],self._motor_angles[11]])
        force = -knee_torque/(0.213*np.cos((np.pi-knee_pos)/2.0))
        force = np.maximum(force,0.0)*9.81/7.7785

        return force

    def footForceBayesian(self, force=None, alpha=0.2):
        bin_num = 11 # 0,12,24,36,48,60,72,84,96,108,120
        foot_force_est = []
        bin = np.array([0,12,24,36,48,60,72,84,96,108,120])
        for i in range(4):
            prior = self.foot_force_prior[i,:]
            likelihood = np.zeros(bin_num)
            for j in range(bin_num):
                likelihood[j] = math.exp(math.log(alpha) * (j*12 / 12.0) ** 2)
            posterior = prior*likelihood
            posterior = posterior/np.sum(posterior)#normalize
            self.foot_force_prior[i,:] = posterior#update prior
            foot_force_est.append(np.sum(bin*posterior))

        return np.array(foot_force_est)

    def GetTimeSinceReset(self):
        return time.time() - self._last_reset_time

    def GetBaseOrientation(self):
        return self._base_orientation.copy()

    def GetFootPositionsInBaseFrame(self):
        """Get the robot's foot position in the base frame."""
        motor_angles = self.GetMotorAngles()
        return foot_positions_in_base_frame(motor_angles)

    def GetHipPositionsInBaseFrame(self):
        """Get the robot's hip position in the base frame."""
        return np.squeeze(HIP_OFFSETS.copy())

    def ComputeJacobian(self, leg_id):
        """Compute the Jacobian for a given leg."""
        # Does not work for Minitaur which has the four bar mechanism for now.
        motor_angles = self.GetMotorAngles()[leg_id * 3:(leg_id + 1) * 3]
        return analytical_leg_jacobian(motor_angles, leg_id)

    def computeBaseAngularVelocity(self):
        foot_positions = self.GetFootPositionsInBaseFrame()
        for leg_id in range(4):
            jacobian = self.ComputeJacobian(leg_id)
            # Only pick the jacobian related to joint motors
            joint_velocities = self.motor_velocities[leg_id * 3:(leg_id + 1) * 3]
            leg_velocity_in_base_frame = jacobian.dot(joint_velocities)
            foot_position = foot_positions[leg_id,:]

    @property
    def motor_velocities(self):
        # return self.GetMotorVelocities()
        return self._raw_motor_velocities.copy()