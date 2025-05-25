"""Estimates base velocity for A1 robot from accelerometer readings."""
import numpy as np
from filterpy.kalman import KalmanFilter
from motion_imitation.utilities.moving_window_filter import MovingWindowFilter
from scipy.spatial.transform import Rotation as R

class VelocityEstimator:
  """Estimates base velocity of A1 robot.

  The velocity estimator consists of 2 parts:
  1) A state estimator for CoM velocity.

  Two sources of information are used:
  The integrated reading of accelerometer and the velocity estimation from
  contact legs. The readings are fused together using a Kalman Filter.

  2) A moving average filter to smooth out velocity readings
  """
  def __init__(self,
               robot,
               accelerometer_variance=0.1,
               sensor_variance=0.1,
               initial_variance=0.1,
               moving_window_filter_size=120,
               imu_offset = [0., 0., 9.91]):
    """Initiates the velocity estimator.

    See filterpy documentation in the link below for more details.
    https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html

    Args:
      robot: the robot class for velocity estimation.
      accelerometer_variance: noise estimation for accelerometer reading.
      sensor_variance: noise estimation for motor velocity reading.
      initial_covariance: covariance estimation of initial state.
    """
    self.robot = robot

    self.filter = KalmanFilter(dim_x=3, dim_z=3, dim_u=3)
    self.filter.x = np.zeros(3)
    self._initial_variance = initial_variance
    self.filter.P = np.eye(3) * self._initial_variance  # State covariance
    self.filter.Q = np.eye(3) * accelerometer_variance
    self.filter.R = np.eye(3) * sensor_variance

    self.filter.H = np.eye(3)  # measurement function (y=H*x)
    self.filter.F = np.eye(3)  # state transition matrix
    self.filter.B = np.eye(3)

    self._window_size = moving_window_filter_size
    self.moving_window_filter_x = MovingWindowFilter(
        window_size=self._window_size)
    self.moving_window_filter_y = MovingWindowFilter(
        window_size=self._window_size)
    self.moving_window_filter_z = MovingWindowFilter(
        window_size=self._window_size)
    self._estimated_velocity = np.zeros(3)
    self._last_timestamp = 0

    self.base_vel_from_foot = np.zeros(3)
    self.imu_offset = np.array(imu_offset)
    self.yaw_vel_from_foot = 0
    self.calibrated_acc = np.zeros(3)
    self.base_vel_from_acc = np.zeros(3)

  def reset(self,imu_offset = [0., 0., 9.91]):
    self.filter.x = np.zeros(3)
    self.filter.P = np.eye(3) * self._initial_variance
    self.moving_window_filter_x = MovingWindowFilter(
        window_size=self._window_size)
    self.moving_window_filter_y = MovingWindowFilter(
        window_size=self._window_size)
    self.moving_window_filter_z = MovingWindowFilter(
        window_size=self._window_size)
    self._last_timestamp = 0
    self.imu_offset = imu_offset

  def _compute_delta_time(self, robot_state):
    if self._last_timestamp == 0.:
      # First timestamp received, return an estimated delta_time.
      delta_time_s = self.robot.time_step
    else:
      delta_time_s = (robot_state.tick - self._last_timestamp) / 1000.
    self._last_timestamp = robot_state.tick
    return delta_time_s

  def update(self, robot_state):
    """Propagate current state estimate with new accelerometer reading."""
    delta_time_s = self._compute_delta_time(robot_state)
    sensor_acc = np.array(robot_state.imu.accelerometer)
    # base_orientation = self.robot.GetBaseOrientation()

    base_orientation_2 = robot_state.imu.quaternion
    # base_orientation_3
    # if (base_orientation == base_orientation_2).all():
    #   print("-------------"*20)
    #   return
    # print("base_orientation:",base_orientation)
    # print("base_orientation2",base_orientation_2)
    # rot_mat = self.robot.pybullet_client.getMatrixFromQuaternion(
    #     base_orientation)
 
    # print(base_orientation)
    if (base_orientation_2 == np.array([0,0,0,0])).all():
      base_orientation_2 = np.array([1,0,0,0])
    # rot_mat_2 = R.from_quat(base_orientation)
    # rot_mat_2 = np.array(rot_mat_2.as_matrix()).reshape((3, 3))
    # print("mat2:",rot_mat_2)
    base_orientation_3 = base_orientation_2
    base_orientation_3[3] = base_orientation_2[0]
    base_orientation_3[:3] = base_orientation_2[1:]
    rot_mat_3 = R.from_quat(base_orientation_3)
    rot_mat_3 = np.array(rot_mat_3.as_matrix()).reshape((3, 3))
    # print("mat3:",rot_mat_3)

    # p = rot_mat_2.as_matrix()
    rot_mat_3 = np.squeeze(rot_mat_3)
    # rot_mat = np.array(rot_mat).reshape((3, 3))
    rot_mat = rot_mat_3
    # print("mat2:",rot_mat_3)
    # print("mat1:", rot_mat)
    # print(rot_mat_2.shape)
    # if (rot_mat == rot_mat_3).all():
    #   print("++++++++++++++++++++++++"*20)
    # else:
    #   print("================"*30)

    # rot_mat = np.array(rot_mat).reshape((3, 3))
    #TODO imu readings need to be correctly offset
    calibrated_acc = rot_mat.dot(sensor_acc) - self.imu_offset#np.array([0., 0., 9.91])#+ np.array([0., 0., -9.8])
    self.calibrated_acc = np.squeeze(calibrated_acc)
    self.filter.predict(u=calibrated_acc * delta_time_s)
    self.base_vel_from_acc = calibrated_acc * delta_time_s
    # print("orn", base_orientation)
    # print("acc", sensor_acc)
    # print("rot_mat", rot_mat)
    # print("predict", self.filter.x[0], self.filter.x[1], self.filter.x[2])

    # Correct estimation using contact legs
    observed_velocities = []
    foot_contact = self.robot.GetFootContacts()

    #variables for estimating yaw
    observed_velocities_in_base_frame = []
    velocity_in_base = np.zeros((4,3))
    foot_position_in_base = self.robot.GetFootPositionsInBaseFrame()
    for leg_id in range(4):
      if foot_contact[leg_id]:
        jacobian = self.robot.ComputeJacobian(leg_id)
        # Only pick the jacobian related to joint motors
        joint_velocities = self.robot.motor_velocities[leg_id *
                                                       3:(leg_id + 1) * 3]
        leg_velocity_in_base_frame = jacobian.dot(joint_velocities)
        base_velocity_in_base_frame = -leg_velocity_in_base_frame[:3]
        #TODO something seems to be wrong with the sign of velocity from foot
        # need to flip left and right leg, joint angles and foot contact
        # base_velocity_in_base_frame = leg_velocity_in_base_frame[:3]
        # base_velocity_in_base_frame[2] = -base_velocity_in_base_frame[2]#flip sign for z
        observed_velocities.append(rot_mat.dot(base_velocity_in_base_frame))
        observed_velocities_in_base_frame.append(np.array(base_velocity_in_base_frame))
        # observed_velocities.append(base_velocity_in_base_frame)

        #estimate yaw velocity from foot
        velocity_in_base[leg_id,:] = np.squeeze(-leg_velocity_in_base_frame)#for calculating yaw
    if observed_velocities:
      observed_velocities = np.mean(observed_velocities, axis=0)
      repeat = 1
      for i in range(repeat):
        self.filter.update(observed_velocities)
      self.base_vel_from_foot = observed_velocities
    # print("update", self.filter.x[0], self.filter.x[1], self.filter.x[2])

    #estimate yaw velocity from foot
    self.yaw_vel_from_foot = 0
    if observed_velocities_in_base_frame :#mbase_velocity_in_base_frameore than one foot in contact
      observed_velocities_in_base_frame = np.mean(observed_velocities_in_base_frame, axis=0)
      yaw_vel = []
      hip_position_in_base = self.robot.GetHipPositionsInBaseFrame()
      for leg_id in range(4):
        if foot_contact[leg_id]:
          vel = (velocity_in_base[leg_id,0:2]-observed_velocities_in_base_frame[0:2])
          pos =  hip_position_in_base[leg_id,0:2]#foot_position_in_base[leg_id,0:2]
          ang_vel = -np.cross(vel,pos)/np.linalg.norm(pos)# match sign of gyro
          ang_vel = ang_vel*4 #TODO yaw vel calculated is around 3~4 times lower than measured, scale up by 4

          # projected_pos = -np.cross(vel, pos) / np.linalg.norm(vel)  # match sign of gyro
          # ang_vel = np.linalg.norm(vel)/projected_pos

          yaw_vel.append(ang_vel)
      self.yaw_vel_from_foot = np.mean(yaw_vel)

    vel_x = self.moving_window_filter_x.calculate_average(self.filter.x[0])
    vel_y = self.moving_window_filter_y.calculate_average(self.filter.x[1])
    vel_z = self.moving_window_filter_z.calculate_average(self.filter.x[2])
    self._estimated_velocity = np.array([vel_x, vel_y, vel_z])

  @property
  def estimated_velocity(self):
    return self._estimated_velocity.copy()

