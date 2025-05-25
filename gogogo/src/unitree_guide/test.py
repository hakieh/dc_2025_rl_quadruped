#!/usr/bin/python

import sys
import time
import math
import numpy as np
#import matplotlib.pyplot as plt
import pickle

import threading

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(currentdir)

parentdir = os.path.dirname(parentdir)
os.sys.path.insert(0, parentdir)
print(parentdir)


from go1_robot_real import *



env = Go1_robot(motor_command_freq=500,control_freq=25)
env.udp.Send()


# while 1:
#     obs = env.receiveObservation()
#     print("self.rx:", env.rx)
#     print("self.ry:", env.ry)


# test stand up 
for i in range(10):
    obs1 = env.receiveObservation()
    ang = env._motor_angles_stand

    print(ang)
    time.sleep(1)
# env.stand(5)
obs1 = env.receiveObservation()

actions = env._motor_angles_stand
print(actions,"======")
# env.rest()
# while 1:
#     tic = time.perf_counter()
#     env.keep_stand()
#     # time.sleep(0.002)

#     toc = time.perf_counter()
#     duration = toc-tic
#     delay = np.clip(env.time_step-duration,0.0,env.time_step)
#     time.sleep(delay)
#     # '''

# while 1:
#     obs1 = env.receiveObservation()
#     obs = env.getObservation_isaacsim()
#     print(obs[:,:45])

#     time.sleep(0.5)
# obs2 = env.getObservation()
# obs3 = env.filterObservation()


# print(obs1)
# print(obs2)
# print(obs3)
