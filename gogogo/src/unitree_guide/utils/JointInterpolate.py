import math
import numpy as np

class JointTrajectoryInterpolate:
    def __init__(self, name=[]):
        self.a=np.zeros(4)
        self.t=0
        self.t_max = 0
        self.name = name

    def interpolate(self, dt):
        self.t = self.t + dt
        self.t = np.clip(self.t, 0.0,self.t_max)#stop interpolation in maximum time step
        target = self.a[3]*(self.t**3) + self.a[2]*(self.t**2) + self.a[1]*(self.t) + self.a[0]

        return target

    def cubic_interpolation_setup(self, q0, dq0, qf, dqf, tf):
        self.a[0] = q0
        self.a[1] = dq0
        self.a[2] = 3.0 * (qf - q0) / tf ** 2 - 2 * dq0 / tf - dqf / tf
        self.a[3] = -2 * (qf - q0) / tf ** 3 + (dqf + dq0) / tf ** 2
        self.t = 0 #reset the timing
        self.t_max = tf #maximum interpolation time
