#!/usr/bin/python

import sys
import time
import math
import numpy as np
import os,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)#os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

print(parentdir+'/lib/python/amd64')
# sys.path.append('../lib/python/amd64')
sys.path.append(parentdir+'/lib/python/amd64')
print(sys.path)

import robot_interface as sdk

def jointLinearInterpolation(initPos, targetPos, rate):

    rate = np.fmin(np.fmax(rate, 0.0), 1.0)
    p = initPos*(1-rate) + targetPos*rate
    return p


if __name__ == '__main__':

    d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
         'FL_0':3, 'FL_1':4, 'FL_2':5, 
         'RR_0':6, 'RR_1':7, 'RR_2':8, 
         'RL_0':9, 'RL_1':10, 'RL_2':11 }
    PosStopF  = math.pow(10,9)
    VelStopF  = 16000.0
    HIGHLEVEL = 0xee
    LOWLEVEL  = 0xff
    sin_mid_q = [0.0, 1.2, -2.0]
    dt = 0.002
    qInit = [0, 0, 0]
    qDes = [0, 0, 0]
    sin_count = 0
    rate_count = 0
    Kp = [0, 0, 0]
    Kd = [0, 0, 0]

    udp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
    safe = sdk.Safety(sdk.LeggedType.Go1)
    
    cmd = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)

    Tpi = 0
    motiontime = 0
    while True:
        dt = 0.1
        time.sleep(dt)
        motiontime += 1

        # print(motiontime)
        # print(state.imu.rpy[0])
        
        
        udp.Recv()
        udp.GetRecv(state)
        
        if( motiontime >= 0):
            motor_angles = np.array([motor.q for motor in state.motorState[:12]])
            print("motor_angles",motor_angles)
            q = state.imu.quaternion
            base_orientation = np.array([q[1], q[2], q[3], q[0]])
            print("quat", base_orientation)
            motor_velocities = np.array([motor.dq for motor in state.motorState[:12]])
            print("motor velocities", motor_velocities)
            base_angular_velocity = np.array(state.imu.gyroscope).copy()
            print("base angular veloxity", base_angular_velocity)
            base_euler = np.array(state.imu.rpy).copy()
            print("base euler", base_euler)


        if(motiontime > 10):
            safe.PowerProtect(cmd, state, 1)

        udp.SetSend(cmd)
        udp.Send()
