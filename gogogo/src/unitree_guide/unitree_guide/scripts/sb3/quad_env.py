import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import rosgraph_msgs.msg as rgm

import torch
import time
import geometry_msgs.msg as gem
import gazebo_msgs.msg as gam
import unitree_legged_msgs.msg as unm
import sensor_msgs.msg as sem
import torch.nn as nn
class QuadEnv(gym.Env):
    
    def __init__(self):
        super(QuadEnv, self).__init__()
        self.ang=gem.Vector3()
        self.vel=gem.Vector3()
        self.pla=gem.Vector3()
        self.ori=gem.Quaternion()
        self.target=gem.Vector3(x=0.7,y=0,z=0)
        print(self.target)
        self.con_time=np.zeros((4),dtype=Float64)
        self.prev_con_time=np.zeros((4),dtype=Float64)
        self.air_time=np.zeros((4),dtype=Float64)
        self.prev_air_time=np.zeros((4),dtype=Float64)
        self.check_con=np.array([True,True,True,True])
        self.j_pos=list(0.0 for i in range(12))
        self.j_vec=list(0.0 for i in range(12))
        self.last_j_vec=list(0.0 for i in range(12))
        self.j_tau=list(0.0 for i in range(12))
        self.j_acc=list(0.0 for i in range(12))
        self.ros_time=0.0
        self.puber=list()
        self.j_sub=list()
        self.count=0

        self.mydiv='cuda'
        max_action = 5
        self.actions=torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0],device=self.mydiv).unsqueeze(0)
        self.dof_map= [1, 4, 7, 10,
                2, 5, 8, 11   ,
                0, 3, 6, 9
                ]
        
        self.rqt_plus=[-1.5,0.1,0.8,-1.5,-0.1,0.8,-1.5,0.1,1,-1.5,-0.1,1]
        self.low_limit=np.array([0.0,-0.2,0.0,-0.2,0.2,0.2,0.2,0.2,-2.3,-2.3,-2.3,-2.3])
        self.up_limit=np.array([0.2, 0.0,0.2, 0.0,1.4,1.4,1.4,1.4,-0.85,-0.85,-0.85,-0.85])
        self.model_plus=torch.tensor(list(self.rqt_plus[self.dof_map[i]] for i in range(0,12)),device=self.mydiv).unsqueeze(0)
        rospy.init_node('quad_rl_env', anonymous=True)
        
        self.action_space = spaces.Box(low=-max_action, high=max_action, shape=(12,), dtype=np.float32)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(45,), dtype=np.float32)
        
        self.puber=self.get_pub()
        self.sub_time=rospy.Subscriber("/clock",rgm.Clock,self.do_time,queue_size=1)
        self.con=np.array([self.ros_time,self.ros_time,self.ros_time,self.ros_time])
        self.air=np.array([self.ros_time,self.ros_time,self.ros_time,self.ros_time])
        self.last_j_vec_time=self.ros_time
        self.sub_imu=rospy.Subscriber("/trunk_imu",sem.Imu,self.do_imu,queue_size=1)
        self.sub_joint=rospy.Subscriber("/go1_gazebo/joint_states",sem.JointState,self.do_joint,queue_size=1)
        self.sub_vel=rospy.Subscriber("/gazebo/model_states",gam.ModelStates,self.do_vel,queue_size=1)
        self.sub_con_FL=rospy.Subscriber("/visual/FL_foot_contact/the_force",gem.WrenchStamped,self.do_con_FL,queue_size=1)
        self.sub_con_FR=rospy.Subscriber("/visual/FR_foot_contact/the_force",gem.WrenchStamped,self.do_con_FR,queue_size=1)
        self.sub_con_RL=rospy.Subscriber("/visual/RL_foot_contact/the_force",gem.WrenchStamped,self.do_con_RL,queue_size=1)
        self.sub_con_RR=rospy.Subscriber("/visual/RR_foot_contact/the_force",gem.WrenchStamped,self.do_con_RR,queue_size=1)
        
        # self.j_sub.append(rospy.Subscriber("/go1_gazebo/FL_hip_controller/state", unm.MotorState, self.do_FLh,queue_size=1))
        # self.j_sub.append(rospy.Subscriber("/go1_gazebo/FR_hip_controller/state", unm.MotorState, self.do_FRh,queue_size=1))
        # self.j_sub.append(rospy.Subscriber("/go1_gazebo/RL_hip_controller/state", unm.MotorState, self.do_RLh,queue_size=1))
        # self.j_sub.append(rospy.Subscriber("/go1_gazebo/RR_hip_controller/state", unm.MotorState, self.do_RRh,queue_size=1))
        # self.j_sub.append(rospy.Subscriber("/go1_gazebo/FL_thigh_controller/state", unm.MotorState, self.do_FLt,queue_size=1))
        # self.j_sub.append(rospy.Subscriber("/go1_gazebo/FR_thigh_controller/state", unm.MotorState, self.do_FRt,queue_size=1))
        # self.j_sub.append(rospy.Subscriber("/go1_gazebo/RL_thigh_controller/state", unm.MotorState, self.do_RLt,queue_size=1))
        # self.j_sub.append(rospy.Subscriber("/go1_gazebo/RR_thigh_controller/state", unm.MotorState, self.do_RRt,queue_size=1))
        # self.j_sub.append(rospy.Subscriber("/go1_gazebo/FL_calf_controller/state", unm.MotorState, self.do_FLc,queue_size=1))
        # self.j_sub.append(rospy.Subscriber("/go1_gazebo/FR_calf_controller/state", unm.MotorState, self.do_FRc,queue_size=1))
        # self.j_sub.append(rospy.Subscriber("/go1_gazebo/RL_calf_controller/state", unm.MotorState, self.do_RLc,queue_size=1))
        # self.j_sub.append(rospy.Subscriber("/go1_gazebo/RR_calf_controller/state", unm.MotorState, self.do_RRc,queue_size=1))
        self.rate=rospy.Rate(50)
        self.rate.sleep()
        self.actions=torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0],device=self.mydiv).unsqueeze(0)
        self.prev_actions=torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0],device=self.mydiv).unsqueeze(0)

        
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

    def do_imu(self,msg_ang):
        self.ang=msg_ang.angular_velocity
        self.ori=msg_ang.orientation
    def do_time(self,msg_time):
        self.ros_time=msg_time.clock.secs+(msg_time.clock.nsecs)/(1e9)
    def do_joint(self,msg_joi):
        
        if self.ros_time-self.last_j_vec_time==0:
            return
        self.last_j_vec=self.j_vec[:]
        for i in range(0,12):
            self.j_pos[i]=msg_joi.position[self.dof_map[i]]
            self.j_vec[i]=msg_joi.velocity[self.dof_map[i]]
            self.j_tau[i]=msg_joi.effort[self.dof_map[i]]
            self.j_acc[i]=(self.j_vec[i]-self.last_j_vec[i])/(self.ros_time-self.last_j_vec_time)
        self.last_j_vec_time=self.ros_time
        
        # self.j_pos=list(msg_joi.position[self.dof_map[i]] for i in range(0,12))
        # self.j_vec=list(msg_joi.velocity[self.dof_map[i]] for i in range(0,12))
        # print(j_pos)
    def do_vel(self,msg_mod):
        self.vel=msg_mod.twist[1].linear
        self.pla=msg_mod.pose[1].position
        # print(self.vel)
        # print(self.pla)
    def do_con_FL(self,msg_con):
        FL_con=msg_con.wrench.force
        f_0=(FL_con.x**2+FL_con.y**2+FL_con.z**2)>50
        if f_0:
            if not self.check_con[0]:
                self.con[0]=self.ros_time
                self.prev_air_time[0]=self.air_time[0]
                self.air_time[0]=0
            else:
                self.con_time[0]=self.ros_time-self.con[0]
        else:
            if self.check_con[0]:
                self.air[0]=self.ros_time
                self.prev_con_time[0]=self.con_time[0]
                self.con_time[0]=0
            else:
                self.air_time[0]=self.ros_time-self.air[0]
        self.check_con[0]=f_0
            
    def do_con_FR(self,msg_con):
        FR_con=msg_con.wrench.force
        f_1=(FR_con.x**2+FR_con.y**2+FR_con.z**2)>50
        if f_1:
            if not self.check_con[1]:
                self.con[1]=self.ros_time
                self.prev_air_time[1]=self.air_time[1]
                self.air_time[1]=0
            else:
                self.con_time[1]=self.ros_time-self.con[1]
        else:
            if self.check_con[1]:
                self.air[1]=self.ros_time
                self.prev_con_time[1]=self.con_time[1]
                self.con_time[1]=0
            else:
                self.air_time[1]=self.ros_time-self.air[1]
        self.check_con[1]=f_1
    def do_con_RL(self,msg_con):
        RL_con=msg_con.wrench.force
        f_2=(RL_con.x**2+RL_con.y**2+RL_con.z**2)>50
        if f_2:
            if not self.check_con[2]:
                self.con[2]=self.ros_time
                self.prev_air_time[2]=self.air_time[2]
                self.air_time[2]=0
            else:
                self.con_time[2]=self.ros_time-self.con[2]
        else:
            if self.check_con[2]:
                self.air[2]=self.ros_time
                self.prev_con_time[2]=self.con_time[2]
                self.con_time[2]=0
            else:
                self.air_time[2]=self.ros_time-self.air[2]
        self.check_con[2]=f_2
    def do_con_RR(self,msg_con):
        RR_con=msg_con.wrench.force
        f_3=(RR_con.x**2+RR_con.y**2+RR_con.z**2)>50
        if f_3:
            if not self.check_con[3]:
                self.con[3]=self.ros_time
                self.prev_air_time[3]=self.air_time[3]
                self.air_time[3]=0
            else:
                self.con_time[3]=self.ros_time-self.con[3]
        else:
            if self.check_con[3]:
                self.air[3]=self.ros_time
                self.prev_con_time[3]=self.con_time[3]
                self.con_time[3]=0
            else:
                self.air_time[3]=self.ros_time-self.air[3]
        self.check_con[3]=f_3

    def get_pub(self):
        puber=list()
        puber.append(rospy.Publisher("/go1_gazebo/FL_calf_controller/command",unm.MotorCmd,queue_size=1))
        puber.append(rospy.Publisher("/go1_gazebo/FL_hip_controller/command",unm.MotorCmd,queue_size=1))
        puber.append(rospy.Publisher("/go1_gazebo/FL_thigh_controller/command",unm.MotorCmd,queue_size=1))
        puber.append(rospy.Publisher("/go1_gazebo/FR_calf_controller/command",unm.MotorCmd,queue_size=1))
        puber.append(rospy.Publisher("/go1_gazebo/FR_hip_controller/command",unm.MotorCmd,queue_size=1))
        puber.append(rospy.Publisher("/go1_gazebo/FR_thigh_controller/command",unm.MotorCmd,queue_size=1))
        puber.append(rospy.Publisher("/go1_gazebo/RL_calf_controller/command",unm.MotorCmd,queue_size=1))
        puber.append(rospy.Publisher("/go1_gazebo/RL_hip_controller/command",unm.MotorCmd,queue_size=1))
        puber.append(rospy.Publisher("/go1_gazebo/RL_thigh_controller/command",unm.MotorCmd,queue_size=1))
        puber.append(rospy.Publisher("/go1_gazebo/RR_calf_controller/command",unm.MotorCmd,queue_size=1))
        puber.append(rospy.Publisher("/go1_gazebo/RR_hip_controller/command",unm.MotorCmd,queue_size=1))
        puber.append(rospy.Publisher("/go1_gazebo/RR_thigh_controller/command",unm.MotorCmd,queue_size=1))
        return puber
        

    def stand_policy(self):
        # print('-----------------------')
        rate0=rospy.Rate(1000)
        targetpos=[-1.3,0.0,0.67,-1.3,0.0,0.67,-1.3,0.0,0.67,-1.3,0.0,0.67]
        kkp=[180,300,180,180,300,180,180,300,180,180,300,180]
        kkd=[8,15,8,8,15,8,8,15,8,8,15,8]
        time.sleep(0.5)
        # print(j_pos)
        startpos=self.j_pos[:]
        duaration=1000
        leg_msg=unm.LowCmd()
        for i in range(12):
            leg_msg.motorCmd[i].mode = 10
            leg_msg.motorCmd[i].Kp = 180
            leg_msg.motorCmd[i].Kd = 8
        percent=float(0)
        # print(puber)
        for i in range(2000):
            percent+=float(1/duaration)
            if percent>1:
                percent=1
            for j in range(12):
                leg_msg.motorCmd[j].q=(1 - percent)*startpos[j] + percent*targetpos[self.dof_map[j]]
                self.puber[self.dof_map[j]].publish(leg_msg.motorCmd[j])
            rate0.sleep()

    def stand_policy_train(self):
        # print('-----------------------')
        rate0=rospy.Rate(50)
        targetpos=[-1.3,0.0,0.67,-1.3,0.0,0.67,-1.3,0.0,0.67,-1.3,0.0,0.67]
        kkp=[180,300,180,180,300,180,180,300,180,180,300,180]
        kkd=[8,15,8,8,15,8,8,15,8,8,15,8]
        # print(j_pos)
        rate0.sleep()
        leg_msg=unm.LowCmd()
        for i in range(12):
            leg_msg.motorCmd[i].mode = 10
            leg_msg.motorCmd[i].Kp = 180
            leg_msg.motorCmd[i].Kd = 8
            leg_msg.motorCmd[i].q=targetpos[self.dof_map[i]]
            self.puber[self.dof_map[i]].publish(leg_msg.motorCmd[i])
        rate0.sleep()

    def get_obs(self):
        velocity_commands=torch.tensor([self.target.x,self.target.y,self.target.z],device=self.mydiv).unsqueeze(0)
        base_ang_vel=torch.tensor([self.ang.x,self.ang.y,self.ang.z],device=self.mydiv).unsqueeze(0)
        quaternion=torch.tensor([self.ori.x,self.ori.y,self.ori.z,self.ori.w],device=self.mydiv).unsqueeze(0)
        projected_gravity = self.quat_rotate_inverse(quaternion)
        joint_pos=torch.tensor(self.j_pos,device=self.mydiv).unsqueeze(0)-self.model_plus
        joint_vec=torch.tensor(self.j_vec,device=self.mydiv).unsqueeze(0)
        obs=torch.cat([base_ang_vel,projected_gravity,velocity_commands,joint_pos,joint_vec,self.actions],dim=-1)
        return obs

    def send(self,output):
        leg_msg=unm.LowCmd()
        # print(output)
        for i in range(12):
            leg_msg.motorCmd[i].mode = 10
            leg_msg.motorCmd[i].Kp = 40
            leg_msg.motorCmd[i].Kd = 0.5
        for i in range(12):
            leg_msg.motorCmd[i].mode=10
            leg_msg.motorCmd[i].q=output[0][i]
            leg_msg.motorCmd[i].dq=0
            leg_msg.motorCmd[i].tau=0
            self.puber[self.dof_map[i]].publish(leg_msg.motorCmd[i])

    
    #reward from gpt
    def calculate_orientation_reward(self):
        q_target = np.array([1, 0, 0, 0])
        q_current = np.array([self.ori.x,self.ori.y,self.ori.z,self.ori.w])
        dot_product = np.dot(q_target, q_current)
        error = 1 - dot_product ** 2
        k = 10  
        orientation_reward = np.exp(-k * error)
        
        return orientation_reward
    def calculate_velocity_reward(self):
        forward_velocity = self.vel.x
        velocity_error = abs(forward_velocity - self.target.x)
        velocity_reward = np.exp(-5 * velocity_error)
        # velocity_reward= 0.1*np.linalg.norm(np.array([self.vel.x,self.vel.y,self.vel.z]))
        if self.vel.y>0.01 or self.vel.z>0.01:
            velocity_reward=-0.1
        
        return velocity_reward
    def calculate_stability_reward(self):
        angular_stability = self.ang.x**2+self.ang.y**2+self.ang.z**2
        stability_reward = np.exp(-angular_stability)
        return stability_reward
    def calculate_step_reward(self):
        s=0.0
        leg=[0,1,4,5,8,9]
        for i in leg:
            s-=abs(self.j_pos[i]-self.j_pos[i+2])
        leg2=[0,2,4,6,8,10]
        for i in leg2:
            s+=abs(self.j_pos[i]-self.j_pos[i+1])
        return s
    def calculate_energy_penalty(self):
        torques = self.get_joint_torques()
        velocities = self.get_joint_velocities()
        
        power = torques * velocities
        energy_consumption = np.sum(np.abs(power))
        return energy_consumption
    def calculate_total_reward(self):
        """
        计算总奖励函数
        """
        R_ori = self.calculate_orientation_reward()
        R_vel = self.calculate_velocity_reward()
        R_stab = self.calculate_stability_reward()
        R_step=self.calculate_step_reward()
        # P_energy = self.calculate_energy_penalty()
        self.w_ori = 1.0
        self.w_vel = 0.5
        self.w_stab = 0.3
        self.w_energy = 0.1
        self.w_step = 0.5
        total_reward = (self.w_ori * R_ori +
                        self.w_vel * R_vel +
                        self.w_stab * R_stab +
                        self.w_step * R_step)
        
        return total_reward
    
    #reward from issacsim
    def track_lin_vel_xy_exp(self):
        lin_vel_error = (self.vel.x-self.target.x)**2 + (self.vel.y-self.target.y)**2
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
        ang_vel_error = (self.ang.z-self.target.z)**2
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
        return (self.vel.z)**2
    '''
    def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """Penalize z-axis base linear velocity using L2 squared kernel."""
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]
        return torch.square(asset.data.root_lin_vel_b[:, 2])
    '''
    def ang_vel_xy_l2(self):
        return (self.ang.x)**2+(self.ang.y)**2
    '''
    def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """Penalize xy-axis base angular velocity using L2 squared kernel."""
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]
        return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    '''
    def joint_torques_l2(self):
        return np.sum(np.square(self.j_tau))
    '''
    def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """Penalize joint torques applied on the articulation using L2 squared kernel.

        NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
        """
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    '''
    def joint_acc_l2(self):
        return np.sum(np.square(self.j_acc))
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
        currently_in_contact = self.con_time > 0.0
        less_than_dt_in_contact = (self.con_time < 0.02 + 0.5)
        first_contact = currently_in_contact*less_than_dt_in_contact
        last_air_time = self.prev_air_time[:]
        reward = np.sum((last_air_time ) * first_contact)
        reward *= np.linalg.norm(np.array([self.target.x,self.target.y])) > 0.1
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
        quaternion=torch.tensor([self.ori.x,self.ori.y,self.ori.z,self.ori.w],device=self.mydiv).unsqueeze(0)
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
        
        out_of_limits = np.clip((self.up_limit-self.j_pos),-np.inf,0.0)
        out_of_limits += np.clip(self.j_pos-self.low_limit,-np.inf,0.0)
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
        return np.square(self.pla.z - 0.35)
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
        track_lin_vel_xy_exp = (1.0)*self.track_lin_vel_xy_exp()
        track_ang_vel_z_exp = (0.5)*self.track_ang_vel_z_exp()
        # -- penalties
        lin_vel_z_l2 = (-2)*self.lin_vel_z_l2()
        ang_vel_xy_l2 = (-0.05)*self.ang_vel_xy_l2()
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
    '''
    def compute_reward(self):
        """Reward terms for the MDP."""

        # -- task
        track_lin_vel_xy_exp = RewTerm(
            func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
        )
        track_ang_vel_z_exp = RewTerm(
            func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
        )
        # -- penalties
        lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
        ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
        dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
        dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
        action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


        feet_air_time = RewTerm(
            func=mdp.feet_air_time,
            weight=0.125,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
                "command_name": "base_velocity",
                "threshold": 0.5,
            },
        )
        undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-3.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
        )
        # -- optional penalties
        flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1)
        dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
        base_height = RewTerm(func=mdp.base_height_l2,weight=-0.5, params={"target_height": 3.5})
    '''
    #reward from 
    def step(self, acc):
        actions=torch.tensor(acc,device=self.mydiv).unsqueeze(0)
        output=actions[:]*0.25+self.model_plus
        # output=torch.clip(output,self.low_limit,self.up_limit)
        # print(output)
        self.prev_actions=self.actions
        self.actions=actions
        self.send(output)
        self.rate.sleep()  
        
        self.count+=1
        observation = self.get_obs() 
        # print("0:",observation)
        reward = self.compute_total_reward()
        if reward<0:
            reward=0
        # reward=self.calculate_total_reward()
        # print(self.check_con)

        done = (self.pla.z < 0.2 or self.count>1000 or self.air_time[2]>1.5 or self.air_time[3]>1.5)
        
        # reward-=done*(500-self.count)
        return observation[0].cpu().numpy(), reward, done, False,{}

    def reset(self, seed=None, options=None):
        self.stand_policy_train()
        self.count=0

        rospy.wait_for_service('/gazebo/set_model_state')
        reset_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        model_state = ModelState()
        model_state.model_name = 'go1_gazebo'
        model_state.pose.position.x = 0
        model_state.pose.position.y = 0
        model_state.pose.position.z = 0.4
        model_state.reference_frame = 'world'
        reset_proxy(model_state)
        # self.stand_policy_train()
        for _ in range(25):
            self.rate.sleep()
        return self.get_obs()[0].cpu().numpy(),{}
        # return np.concatenate([self.joint_positions, self.joint_velocities])
