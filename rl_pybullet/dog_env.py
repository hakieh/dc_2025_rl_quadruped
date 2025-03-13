import pybullet as p
import pybullet_envs
import pybullet_data
import torch 
import gymnasium as gym
from gymnasium import spaces
import numpy as np 
import math
import os
import inv_kine.inv_kine as ik
import time
class TestudogEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(TestudogEnv, self).__init__()
        self.step_dt=0.01
        self.init_state()
        self.action_space = spaces.Box(low=-5, high=5, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(45,), dtype=np.float32)
        self.dof_map=[9,2,23,16,10,3,24,17,11,4,25,18]
        self.target=[0.7,0,0]
        self.mydiv='cuda'
        self.rqt_plus=[0.1,-0.1,0.1,-0.1,0.8,0.8,1,1,-1.5,-1.5,-1.5,-1.5]
        # self.low_limit=torch.tensor([0.0,-0.2,0.0,-0.2,0.4,0.4,0.4,0.4,-2.3,-2.3,-2.3,-2.3],device=self.mydiv).unsqueeze(0)
        # self.up_limit=torch.tensor( [0.2, 0.0,0.2, 0.0,1.4,1.4,1.4,1.4,-0.85,-0.85,-0.85,-0.85],device=self.mydiv).unsqueeze(0)
        self.low_limit=np.array([0.0,-0.2,0.0,-0.2,0.4,0.4,0.4,0.4,-2.3,-2.3,-2.3,-2.3])
        self.up_limit=np.array([0.2, 0.0,0.2, 0.0,1.4,1.4,1.4,1.4,-0.85,-0.85,-0.85,-0.85])
        self.model_plus=torch.tensor(list(self.rqt_plus[i] for i in range(0,12)),device=self.mydiv).unsqueeze(0)
        self.actions=torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0],device=self.mydiv).unsqueeze(0)
        self.prev_actions=torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0],device=self.mydiv).unsqueeze(0)
        self.con_time=np.zeros((4),dtype=np.float64)
        self.prev_con_time=np.zeros((4),dtype=np.float64)
        self.air_time=np.zeros((4),dtype=np.float64)
        self.prev_air_time=np.zeros((4),dtype=np.float64)
        self.check_con=np.array([True,True,True,True])
        self.con=np.array([time.time(),time.time(),time.time(),time.time()])
        self.air=np.array([time.time(),time.time(),time.time(),time.time()])
        self._last_base_position    = [0, 0, 0]
        self._last_base_orientation = [0, 0, 0]
        self._last_joint_positions  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._last_joint_velocities = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        
    def init_state(self):
        self.count = 0
        p.connect(p.DIRECT)
        # p.connect(p.GUI)
        p.resetSimulation()
        p.setGravity(0,0,-9.8)
        # p.setTimeStep(self.step_dt)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf",[0,0,0],[0,0,0,1])
        self.testudogid = p.loadURDF("./urdf/go1.urdf",[0,0,0.4],[0,0,0,1])
        # p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 0])
        # focus,_ = p.getBasePositionAndOrientation(self.testudogid)
        # p.resetDebugVisualizerCamera()
        
        # # observation --> body: pos, rot, lin_vel ang_vel / joints: pos, vel / foot position? / foot contact?
        # body_pos = p.getLinkState(self.testudogid,0)[0]
        # body_rot = p.getLinkState(self.testudogid,0)[1]
        # body_rot_rpy = p.getEulerFromQuaternion(body_rot) 
        # body_lin_vel = p.getLinkState(self.testudogid,0,computeLinkVelocity=1)[6]
        # body_ang_vel = p.getLinkState(self.testudogid,0,computeLinkVelocity=1)[7]
        # joint_pos = []
        # joint_vel = []
        # joint_torque = []
        # for i in range(12):
        #     joint_pos.append(p.getJointState(self.testudogid,self.dof_map[i])[0])
        #     joint_vel.append(p.getJointState(self.testudogid,self.dof_map[i])[1])        
        #     joint_torque.append(p.getJointState(self.testudogid,self.dof_map[i])[3]) 
        # obs = list(body_pos) + list(body_rot_rpy)[0:2] + list(body_lin_vel) + list(body_ang_vel) + joint_pos + joint_vel + joint_torque
    
    def quat_rotate_inverse(self,q):
        shape=q.shape
        v=torch.tensor([0,0,-1],dtype=torch.float,device= self.mydiv).unsqueeze(0)
        q_w = q[:, -1]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * \
            torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
                shape[0], 3, 1)).squeeze(-1) * 2.0
        return a - b + c
    def get_obs(self):
        ang = p.getLinkState(self.testudogid,0,computeLinkVelocity=1)[7]
        ori = p.getLinkState(self.testudogid,0)[1]
        j_pos=[]
        j_vec=[]
        for i in range(0,12):
            j_pos.append(p.getJointState(self.testudogid,self.dof_map[i])[0])
            j_vec.append(p.getJointState(self.testudogid,self.dof_map[i])[1])
        velocity_commands=torch.tensor([self.target[0],self.target[1],self.target[2]],device=self.mydiv).unsqueeze(0)
        base_ang_vel=torch.tensor([ang[0],ang[1],ang[2]],device=self.mydiv).unsqueeze(0)
        quaternion=torch.tensor([ori[0],ori[1],ori[2],ori[3]],device=self.mydiv).unsqueeze(0)
        projected_gravity = self.quat_rotate_inverse(quaternion)
        joint_pos=torch.tensor(j_pos,device=self.mydiv).unsqueeze(0)-self.model_plus
        joint_vec=torch.tensor(j_vec,device=self.mydiv).unsqueeze(0)
        obs=torch.cat([base_ang_vel,projected_gravity,velocity_commands,joint_pos,joint_vec,self.actions],dim=-1)
        return obs

    def reset(self, seed=None, options=None):
        p.disconnect()
        self.init_state()
        obs=self.get_obs()
        targetpos=[0,0,0,0,0.67,0.67,0.67,0.67,-1.3,-1.3,-1.3,-1.3]
        p.setJointMotorControlArray(self.testudogid,self.dof_map,p.POSITION_CONTROL,
            targetPositions=targetpos)
        p.stepSimulation()
        return obs[0].cpu().numpy(),{}
        
    def step(self,acc):
        actions=torch.tensor(acc,device=self.mydiv).unsqueeze(0)
        output=actions[:]*0.25+self.model_plus
        # output=torch.clip(output,self.low_limit,self.up_limit)
        self.prev_actions=self.actions
        self.actions=actions

        p.setJointMotorControlArray(self.testudogid,self.dof_map,p.POSITION_CONTROL,
            targetPositions=output[0].cpu().numpy())
        
        p.stepSimulation()
        # time.sleep(self.dt)
        reward=self.compute_total_reward()
        if(reward<0):
            reward=0
        # reward=self.crawling_compute_reward()
        self.count+=1
        # terminal fail condition eg robot fall  
        # survival reward
        done = (self.check_fall() or self.count>1000)
        
        
        obs=self.get_obs()
        return obs[0].cpu().numpy(), reward, done, False,{} 

    def check_fall(self):
        orientation = p.getLinkState(self.testudogid,0)[1]
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)

        # print("RPY: ", roll, pitch, yaw)
        
        # Check if the robot has fallen over (adjust the threshold as needed)
        # fallen when 0 > roll > pi or -pi/2 < pitch > pi/2 
        # if (np.pi/12) > roll or roll > ((11/12)*np.pi) or (-5/12)*np.pi > pitch  or pitch > (5/12)*np.pi:
        return (roll < -np.pi/2 or roll > (2/2)*np.pi or pitch < -0.8 or pitch > 0.8 or p.getLinkState(self.testudogid,0)[0][2]<0.2)
    
    def track_lin_vel_xy_exp(self):
        vel=p.getLinkState(self.testudogid,0,computeLinkVelocity=1)[6]
        lin_vel_error = (vel[0]-self.target[0])**2 + (vel[1]-self.target[1])**2
        return np.exp(-lin_vel_error * 4)
    '''
    def track_lin_vel_xy_exp(
        env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
     ) -> torch.Tensor:
        """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]
        # compute the error
        lin_vel_error = (torch.sum(
            torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
            dim=1,
        ))
        return torch.exp(-lin_vel_error / std**2)
    '''
    def track_ang_vel_z_exp(self):
        ang = p.getLinkState(self.testudogid,0,computeLinkVelocity=1)[7]
        ang_vel_error = (ang[2]-self.target[2])**2
        return np.exp(-ang_vel_error *4)
    '''
    def track_ang_vel_z_exp(
        env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]
        # compute the error
        ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
        return torch.exp(-ang_vel_error / std**2)
    '''
    def lin_vel_z_l2(self):
        vel=p.getLinkState(self.testudogid,0,computeLinkVelocity=1)[6]
        return (vel[2])**2
    '''
    def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """Penalize z-axis base linear velocity using L2 squared kernel."""
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]
        return torch.square(asset.data.root_lin_vel_b[:, 2])
    '''
    def ang_vel_xy_l2(self):
        ang = p.getLinkState(self.testudogid,0,computeLinkVelocity=1)[7]
        return (ang[0])**2+(ang[1])**2
    '''
    def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """Penalize xy-axis base angular velocity using L2 squared kernel."""
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]
        return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    '''
    def joint_torques_l2(self):
        j_tau=[]
        for i in range(0,12):
            j_tau.append(p.getJointState(self.testudogid,self.dof_map[i])[3])
        return np.sum(np.square(j_tau))
    '''
    def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """Penalize joint torques applied on the articulation using L2 squared kernel.

        NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
        """
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    '''

    '''
    def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """Penalize joint accelerations on the articulation using L2 squared kernel.

        NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
        """
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)
    '''
    def action_rate_l2(self):
        return np.sum(np.square(self.actions[0].cpu().numpy() - self.prev_actions[0].cpu().numpy()))
    '''
    def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
        """Penalize the rate of change of the actions using L2 squared kernel."""
        return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    '''
    def feet_air_time(self):
        f_0=p.getContactPoints(bodyA=self.testudogid,linkIndexA=12) != ()
        f_1=p.getContactPoints(bodyA=self.testudogid,linkIndexA=5) != ()
        f_2=p.getContactPoints(bodyA=self.testudogid,linkIndexA=26) != ()
        f_3=p.getContactPoints(bodyA=self.testudogid,linkIndexA=19) != ()
        if f_0:
            if not self.check_con[0]:
                self.con[0]=time.time()
                self.prev_air_time[0]=self.air_time[0]
                self.air_time[0]=0
            else:
                self.con_time[0]=time.time()-self.con[0]
        else:
            if self.check_con[0]:
                self.air[0]=time.time()
                self.prev_con_time[0]=self.con_time[0]
                self.con_time[0]=0
            else:
                self.air_time[0]=time.time()-self.air[0]
        self.check_con[0]=f_0
        if f_1:
            if not self.check_con[1]:
                self.con[1]=time.time()
                self.prev_air_time[1]=self.air_time[1]
                self.air_time[1]=0
            else:
                self.con_time[1]=time.time()-self.con[1]
        else:
            if self.check_con[1]:
                self.air[1]=time.time()
                self.prev_con_time[1]=self.con_time[1]
                self.con_time[1]=0
            else:
                self.air_time[1]=time.time()-self.air[1]
        self.check_con[1]=f_1
        if f_2:
            if not self.check_con[2]:
                self.con[2]=time.time()
                self.prev_air_time[2]=self.air_time[2]
                self.air_time[2]=0
            else:
                self.con_time[2]=time.time()-self.con[2]
        else:
            if self.check_con[2]:
                self.air[2]=time.time()
                self.prev_con_time[2]=self.con_time[2]
                self.con_time[2]=0
            else:
                self.air_time[2]=time.time()-self.air[2]
        self.check_con[2]=f_2
        if f_3:
            if not self.check_con[3]:
                self.con[3]=time.time()
                self.prev_air_time[3]=self.air_time[3]
                self.air_time[3]=0
            else:
                self.con_time[3]=time.time()-self.con[3]
        else:
            if self.check_con[3]:
                self.air[3]=time.time()
                self.prev_con_time[3]=self.con_time[3]
                self.con_time[3]=0
            else:
                self.air_time[3]=time.time()-self.air[3]
        self.check_con[3]=f_3
        currently_in_contact = self.con_time > 0.0
        less_than_dt_in_contact = (self.con_time < 0.52)
        first_contact = currently_in_contact*less_than_dt_in_contact
        last_air_time = self.prev_air_time[:]
        reward = np.sum((last_air_time ) * first_contact)
        reward *= np.linalg.norm(np.array([self.target[0],self.target[1]])) > 0.1
        return reward
    '''
    def feet_air_time(
        env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
    ) -> torch.Tensor:
        """Reward long steps taken by the feet using L2-kernel.

        This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
        that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
        the time for which the feet are in the air.

        If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
        """
        # extract the used quantities (to enable type-hinting)
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        # compute the reward
        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
        # no reward for zero command
        reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
        return reward
    '''
    
    '''
    def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        """Penalize undesired contacts as the number of violations that are above a threshold."""
        # extract the used quantities (to enable type-hinting)
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        # check if contact force is above threshold
        net_contact_forces = contact_sensor.data.net_forces_w_history
        is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
        # sum over contacts for each environment
        return torch.sum(is_contact, dim=1)
    '''
    def flat_orientation_l2(self):
        ori = p.getLinkState(self.testudogid,0)[1]
        quaternion=torch.tensor([ori[0],ori[1],ori[2],ori[3]],device=self.mydiv).unsqueeze(0)
        projected_gravity = self.quat_rotate_inverse(quaternion)
        return np.sum(np.square(projected_gravity[:, :2].cpu().numpy()))
    '''
    def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """Penalize non-flat base orientation using L2 squared kernel.

        This is computed by penalizing the xy-components of the projected gravity vector.
        """
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]
        return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    ''' 
    def joint_pos_limits(self):
        j_pos=[]
        for i in range(0,12):
            j_pos.append(p.getJointState(self.testudogid,self.dof_map[i])[0])
        out_of_limits = np.clip((self.up_limit-j_pos),-np.inf,0.0)
        out_of_limits += np.clip(j_pos-self.low_limit,-np.inf,0.0)
        return np.sum(out_of_limits)
    '''
    def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """Penalize joint positions if they cross the soft limits.

        This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
        """
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute out of limits constraints
        out_of_limits = -(
            asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
        ).clip(max=0.0)
        out_of_limits += (
            asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
        ).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)
    '''
    def base_height_l2(self):
        pos = p.getLinkState(self.testudogid,0)[0]
        return np.square(pos[2] - 0.35)
    '''
        def base_height_l2(
        env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        """Penalize asset height from its target using L2 squared kernel.

        Note:
            Currently, it assumes a flat terrain, i.e. the target height is in the world frame.
        """
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]
        time_mask = env.episode_length_buf>100
        # print(env.episode_length_buf[0])
        # TODO: Fix this for rough-terrain.
        return torch.square(asset.data.root_pos_w[:, 2] - target_height)*time_mask
    '''
    def compute_total_reward(self):
        # self.states=p.getJointStates(self.testudogid,self.dof_map)
        track_lin_vel_xy_exp = (1.0)*self.track_lin_vel_xy_exp()
        track_ang_vel_z_exp = (0.5)*self.track_ang_vel_z_exp()
        # -- penalties
        lin_vel_z_l2 = (-0.2)*self.lin_vel_z_l2()
        ang_vel_xy_l2 = (-0.01)*self.ang_vel_xy_l2()
        dof_torques_l2 = (-1.0e-5)*self.joint_torques_l2()
        # dof_acc_l2 = (-2.5e-8)*self.joint_acc_l2()
        action_rate_l2 = (-0.01)*self.action_rate_l2()
        feet_air_time = (1.0)*self.feet_air_time()
        # -- optional penalties
        flat_orientation_l2 = (-1)*self.flat_orientation_l2()
        dof_pos_limits=(1)*self.joint_pos_limits()
        base_height = (-0.5)*self.base_height_l2()

        # print("1",track_lin_vel_xy_exp)
        # print("2",track_ang_vel_z_exp)
        # print("3",lin_vel_z_l2)
        # print("4",ang_vel_xy_l2)
        # print("5",dof_torques_l2)
        # print("6",action_rate_l2)
        # print("7",feet_air_time)
        # print("8",flat_orientation_l2)
        # print("9",dof_pos_limits)
        # print("10",base_height)
        # print()

        return track_lin_vel_xy_exp + track_ang_vel_z_exp + lin_vel_z_l2 + ang_vel_xy_l2 + dof_torques_l2  + action_rate_l2 + feet_air_time + flat_orientation_l2 + dof_pos_limits + base_height
        # return track_lin_vel_xy_exp + track_ang_vel_z_exp + feet_air_time 

    def crawling_compute_reward(self):

        # ToDo
        # Negative Reward for joints_at limit
        # Negative Reward for joint_collisions
        # Negative Reward for side to side motion
        # Negative Reward for body rotation

        self._distance_weight = 1.0 # To use in future when having a goal target position
        self._energy_weight = 0.005 # typical energy values [0.01,0.1]
        self._shake_weight = 0.0 #typical shake values [0.00001,0.001]
        self._drift_weight = 0.0
        self._velocity_weight = .10 #typical velocity values [0.01,0.50]
        self._jitter_weight = 0.1
    
        # Get the linear velocity of the base link of the robot
        # Calculate the magnitude (absolute value) of the linear velocity
        # linear_velocity, angular_velocity = p.getBaseVelocity(self.robot_id)
        linear_velocity=p.getLinkState(self.testudogid,0,computeLinkVelocity=1)[6]
        angular_velocity = p.getLinkState(self.testudogid,0,computeLinkVelocity=1)[7]

        # velocity_magnitude = np.linalg.norm(linear_velocity) # for bounding gait I used the x,y,z components of velocity
        velocity_magnitude = np.linalg.norm(linear_velocity[:2]) # Use only x and y components of the linear velocity

        # The reward is the magnitude of the base link's velocity
        # self.current_base_position, self.current_base_orientation = p.getBasePositionAndOrientation(self.testudogid)
        self.current_base_position=p.getLinkState(self.testudogid,0)[0]
        self.current_base_orientation=p.getLinkState(self.testudogid,0)[1]

        forward_reward = self.current_base_position[0] - self._last_base_position[0]
        drift_reward = -abs(self.current_base_position[1] - self._last_base_position[1]) # negative to reduce drift (yf-yo)
        shake_reward = -abs(self.current_base_position[2] - self._last_base_position[2]) # negative to reduce shaking (zf-zo)

        self._last_base_position = self.current_base_position

        joint_states = p.getJointStates(self.testudogid, self.dof_map)
        self.current_joint_positions = [state[0] for state in joint_states] #1x8 
        self.current_joint_torques = [state[3] for state in joint_states]
        self.current_joint_velocities = [state[1] for state in joint_states]

        jitter_penalty = -sum(abs(np.array(self.current_joint_velocities) - np.array(self._last_joint_velocities)))

        self._last_joint_positions  = self.current_joint_positions
        self._last_joint_velocities = self.current_joint_velocities
        
        
        energy_reward = abs(np.dot(self.current_joint_torques, self.current_joint_velocities))  * self.step_dt # Negative to penalize energy consumption E = (T*dq)/(delta(t))

        # Penalize falling 
        fall_weight = -1
        fall_penalty = fall_weight if self.check_fall() else 0

        # Penalize high angular velocities
        angular_penalty = -np.linalg.norm(angular_velocity)

        # Debugging Rewards
        # print('Velocity_magnitude: ', velocity_magnitude)
        # print('Energy reward: -', energy_reward)
        # print('Shake reward: ', shake_reward)
        # print('Fall Penalty: ', fall_penalty)

        target_velocity = 0.1 #m/s
        epsilon = 0.000001 #small number
        velocity_reward = self._velocity_weight * (1 / (abs(target_velocity - velocity_magnitude) + epsilon))
        

        # Copy of Minitaur reward with fall penalty added

        reward = (self._distance_weight * forward_reward 
                  - self._energy_weight * energy_reward 
                  + self._drift_weight * drift_reward 
                  + self._shake_weight * shake_reward
                  + fall_penalty #Penalty for falling
                  )


        # This is the reward I used during 811
        # reward = (self._velocity_weight*velocity_magnitude # Reward for robot moving fast
        #           - self._energy_weight * energy_reward # Penality for motor consumption
        #           + self._shake_weight * shake_reward # Penalty for vertical shaking
        #           + fall_penalty #Penalty for falling
        #           )
        
        #print(self.current_step, '  :', reward)
        
        # if fall_penalty != 0:
        #     print(self.current_step, '  :', reward)

        

        #No distance weight - Bounding

        # reward = (self._velocity_weight*velocity_magnitude - self._energy_weight * energy_reward +
        #       self._drift_weight * drift_reward + self._shake_weight * shake_reward)
        
        # Distance weights
        # reward = (self._velocity_weight*velocity_magnitude + self._distance_weight * forward_reward - self._energy_weight * energy_reward +
        #       self._drift_weight * drift_reward + self._shake_weight * shake_reward)


        #reward =  velocity_magnitude + fall_penalty + angular_penalty # Testing if negative velocity the robot should come to a stop from training
        # print("REWARD:", reward)
        #reward = 0 
        return reward