# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pybullet simulation of a Laikago robot."""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import re
import numpy as np

# #A1
#
# COM_OFFSET = -np.array([0.012731, 0.002186, 0.000515])
# HIP_OFFSETS = np.array([[0.183, -0.047, 0.], [0.183, 0.047, 0.],
#                         [-0.183, -0.047, 0.], [-0.183, 0.047, 0.]
#                         ]) + COM_OFFSET

#Go1

COM_OFFSET = -np.array([-0.01592 -0.06659 -0.00617])
HIP_OFFSETS = np.array([[0.1881, -0.04675, 0.], [0.1881, 0.04675, 0.],
                        [-0.1881, -0.04675, 0.], [-0.1881, 0.04675, 0.]
                        ]) + COM_OFFSET

def foot_position_in_hip_frame_to_joint_angle(foot_position, l_hip_sign=1):
  # #A1
  # l_up = 0.2
  # l_low = 0.2
  # l_hip = 0.08505 * l_hip_sign

  #Go1
  l_up = 0.213
  l_low = 0.213
  l_hip = 0.08 * l_hip_sign
  x, y, z = foot_position[0], foot_position[1], foot_position[2]
  theta_knee = -np.arccos(
      (x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) /
      (2 * l_low * l_up))
  l = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(theta_knee))
  theta_hip = np.arcsin(-x / l) - theta_knee / 2
  c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
  s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
  theta_ab = np.arctan2(s1, c1)
  return np.array([theta_ab, theta_hip, theta_knee])


def foot_position_in_hip_frame(angles, l_hip_sign=1):
  theta_ab, theta_hip, theta_knee = angles[0], angles[1], angles[2]
  # l_up = 0.2
  # l_low = 0.2
  # l_hip = 0.08505 * l_hip_sign

  #Go1
  l_up = 0.213
  l_low = 0.213
  l_hip = 0.08 * l_hip_sign
  leg_distance = np.sqrt(l_up**2 + l_low**2 +
                         2 * l_up * l_low * np.cos(theta_knee))
  eff_swing = theta_hip + theta_knee / 2

  off_x_hip = -leg_distance * np.sin(eff_swing)
  off_z_hip = -leg_distance * np.cos(eff_swing)
  off_y_hip = l_hip

  # off_x = off_x_hip
  # off_y = np.cos(theta_ab) * off_y_hip - np.sin(theta_ab) * off_z_hip
  # off_z = np.sin(theta_ab) * off_y_hip + np.cos(theta_ab) * off_z_hip
  # hip abduction joint configuration different between erwin motion_imitation and go1 official
  off_x = off_x_hip
  off_y = np.cos(theta_ab) * off_y_hip - np.sin(theta_ab) * off_z_hip
  off_z = np.sin(theta_ab) * off_y_hip + np.cos(theta_ab) * off_z_hip
  return np.array([off_x, off_y, off_z])


def analytical_leg_jacobian(leg_angles, leg_id):
  """
  Computes the analytical Jacobian.
  Args:
  ` leg_angles: a list of 3 numbers for current abduction, hip and knee angle.
    l_hip_sign: whether it's a left (1) or right(-1) leg.
  """
  # l_up = 0.2
  # l_low = 0.2
  # l_hip = 0.08505 * (-1)**(leg_id + 1)

  #Go1
  l_up = 0.213
  l_low = 0.213
  l_hip = 0.08 * (-1)**(leg_id + 1)

  t1, t2, t3 = leg_angles[0], leg_angles[1], leg_angles[2]
  l_eff = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(t3))
  t_eff = t2 + t3 / 2
  J = np.zeros((3, 3))
  J[0, 0] = 0
  J[0, 1] = -l_eff * np.cos(t_eff)
  J[0, 2] = l_low * l_up * np.sin(t3) * np.sin(t_eff) / l_eff - l_eff * np.cos(
      t_eff) / 2
  J[1, 0] = -l_hip * np.sin(t1) + l_eff * np.cos(t1) * np.cos(t_eff)
  J[1, 1] = -l_eff * np.sin(t1) * np.sin(t_eff)
  J[1, 2] = -l_low * l_up * np.sin(t1) * np.sin(t3) * np.cos(
      t_eff) / l_eff - l_eff * np.sin(t1) * np.sin(t_eff) / 2
  J[2, 0] = l_hip * np.cos(t1) + l_eff * np.sin(t1) * np.cos(t_eff)
  J[2, 1] = l_eff * np.sin(t_eff) * np.cos(t1)
  J[2, 2] = l_low * l_up * np.sin(t3) * np.cos(t1) * np.cos(
      t_eff) / l_eff + l_eff * np.sin(t_eff) * np.cos(t1) / 2
  return J


# # For JIT compilation
# foot_position_in_hip_frame_to_joint_angle(np.random.uniform(size=3), 1)
# foot_position_in_hip_frame_to_joint_angle(np.random.uniform(size=3), -1)


def foot_positions_in_base_frame(foot_angles):
  foot_angles = foot_angles.reshape((4, 3))
  foot_positions = np.zeros((4, 3))
  for i in range(4):
    foot_positions[i] = foot_position_in_hip_frame(foot_angles[i],
                                                   l_hip_sign=(-1)**(i + 1))
  return foot_positions + HIP_OFFSETS