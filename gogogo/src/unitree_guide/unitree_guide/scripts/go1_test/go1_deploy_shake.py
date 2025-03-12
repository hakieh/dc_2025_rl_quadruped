#!/usr/bin/python

import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle

import threading
import pandas as pd

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(currentdir)

parentdir = os.path.dirname(parentdir)
os.sys.path.insert(0, parentdir)
print(parentdir)

file_id=time.strftime("%Y-%m-%d", time.localtime())
time_id=time.strftime("%H:%M", time.localtime())
print(file_id)
#sys.path.append('../lib/python/amd64')

from go1_robot_real import *

# from rsl_rl.modules import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from himloco.get_actor import get_actor

policy_action_global = np.zeros(12)
observed_state_global = np.zeros(38)
button_global = np.zeros(16)
flag_global = 0
stand_flag_global = 0
#global logging variables
control_action_log = 0
filtered_action_log = 0
observed_state_log = 0

base_euler_log = 0
base_orn_vel_log = 0
base_pos_vel_yaw_log = 0
base_pos_vel_log = 0
base_euler_from_quat_log = 0
base_vel_from_foot_log = 0
base_gravity_vector_log = 0
wireless_remote_log = 0
motor_angle_log = 0
motor_velocity_log = 0
motor_torque_log = 0
base_orn_vel_fusion_log = 0
yaw_vel_from_foot_log = 0
foot_force_log = 0
base_imu_log = 0
calibrated_acc_log = 0
lllen=0

actions_his = np.zeros([1,12]) # log dof
dof_pos_his = np.zeros([1,12])
dof_vel_his = np.zeros([1,12])
tau_his = np.zeros([1,12])

actions_no_his=torch.zeros([1,12])
actions_yes_his=torch.zeros([1,12])
cca_his=torch.zeros([1,12])

class Estimator(nn.Module): 
    def __init__(self):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Linear(225,512),
            nn.ELU(),
            nn.Linear(512,256),
            nn.ELU(),
            nn.Linear(256,128),
            nn.ELU(),
            nn.Linear(128,64),
            nn.ELU(),
            nn.Linear(64,10),
        )
    def forward(self, x):
        # parts = self.encoder(x) 
        # vel,latent = parts[:3],parts[3:]
        # latent = F.normalize(latent,dim=-1,p=2)

        return self.estimator(x)


class Actor(nn.Module): 
    def __init__(self):
        super(Actor, self).__init__()
        self.actor_backbone = nn.Sequential(
            nn.Linear(55,1024),
            nn.ELU(),
            nn.Linear(1024,512),
            nn.ELU(),
            nn.Linear(512,256),
            nn.ELU(),
            nn.Linear(256,128),
            nn.ELU(),
            nn.Linear(128,64),
            nn.ELU(),
            nn.Linear(64,32),
            nn.ELU(),
            nn.Linear(32,16)
        )
        self.actor_backbone2 = nn.Sequential(nn.Linear(16,12))
        # self.estimator = Estimator()
    def forward(self, x):
        # vel,latent = self.estimator(x[0,49:])
        # x = torch.cat((x[0,:49],vel,latent),dim=-1)
        x = self.actor_backbone(x)
        return self.actor_backbone2(x) 
class alg(nn.Module):
    def __init__(self):
        super(alg, self).__init__()
        self.actor = Actor()
    def forward(self, x):
        return self.actor(x)



class Run():
    def __del__(self):

        return

    def __init__(self, config=None, dir_path=None):
        #initialize
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.dir_path = currentdir

        self.run_time = 60*10

        
        # running frequency 
        # self.motor_freq = 500.0
        self.motor_freq = 500.0
        self.motor_dt = 1.0 / self.motor_freq
        self.control_freq = 50.0

        self.control_dt = 1.0 / self.control_freq

        self.env = Go1_robot(motor_command_freq=self.motor_freq,control_freq=self.control_freq)
        
        time.sleep(5)

        return

    def quat_rotate_inverse(q, v): # [x,y,z,w]
        shape = q.shape
        q_w = q[:, -1]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * \
            torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
                shape[0], 3, 1)).squeeze(-1) * 2.0
        return a - b + c

    def get_actor3(self):
        actor = Actor()
        path = "./weights/model_3500.pt"
        loaded  = torch.load(path,map_location="cpu")
        actor.load_state_dict(loaded['model_state_dict'],strict=False)
        actor.to(torch.device("cuda"))
        # depth_actor = actor_critic.act_inference
        # rospy.loginfo("------------loading action model successfully----------------")
        print(actor)
        return actor

    def get_actor4(self):
        loaded = torch.load("/home/zby/fall_recover/go1_deployment_2/shake/model_13000.pt",map_location="cpu")
        # print(1)
        actor = alg()
        estimator = Estimator()

        actor.load_state_dict(loaded['model_state_dict'],strict=False)
        actor.to(torch.device("cuda"))

        estimator.load_state_dict(loaded['estimator_state_dict'])
        estimator.to(torch.device("cuda"))

        # a = torch.ones((1,225),device="cuda:0")
        # print(estimator(a))
        # b = torch.ones((1,55),device="cuda:0")
        # print(actor(b))
        return actor,estimator

    def policy(self, lock):
        global observed_state_global
        global policy_action_global
        global button_global
        global flag_global
        global stand_flag_global
        #log variables
        global control_action_log
        global filtered_action_log
        global observed_state_log

        global base_euler_log
        global base_orn_vel_log
        global base_pos_vel_yaw_log
        global base_pos_vel_log
        global base_euler_from_quat_log
        global base_vel_from_foot_log
        global base_gravity_vector_log
        global wireless_remote_log
        global motor_angle_log
        global motor_velocity_log
        global motor_torque_log
        global base_orn_vel_fusion_log
        global yaw_vel_from_foot_log
        global foot_force_log
        global base_imu_log
        global calibrated_acc_log
        global lllen
        global actions_his # log dof
        global dof_pos_his
        global dof_vel_his
        global tau_his
        init_flag = 1
        record_angle = {}
        record_angle["angle"] = []
        record_angle["real"] = []
        policy_duration = 0

        #logging
        base_pos_vel_yaw_list = []
        base_orn_vel_list = []
        base_pos_vel_list = []
        base_euler_list = []
        base_euler_from_quat_list = []
        base_vel_from_foot_list = []
        base_gravity_vector_list = []
        control_action_list = []
        filtered_action_list = []
        observed_state_list = []
        wireless_remote_list = []
        motor_velocity_list = []
        motor_torque_list = []
        motor_angle_list = []
        yaw_vel_from_foot_list = []
        base_orn_vel_fusion_list = []
        foot_force_list = []
        base_imu_list = []
        calibrated_acc_list = []

        self.actor,self.estimator = self.get_actor4()

        j=0
        while 1:
            lock.acquire()
            flag = flag_global
            lock.release()
            time.sleep(0.5)
            if flag != 0:
                break
        # todo
        # observed_state = np.nan_to_num(np.squeeze(observed_state_global), copy=True, nan=0)  # replace nan with 0
        # state = np.squeeze(observed_state)
        state = observed_state_global
        # 测试 actor
        # action = self.agent.action(state)
        # action = self.actor(state)
        print("agent loaded, successfully generated action")

        
        duara_his=np.zeros([1,1])
        init_motor_angle =torch.tensor([self.env._raw_motor_angles[self.env.dof_map_isaacsim[i]] for i in range(12)],device=self.env.model_device,dtype=torch.float32).unsqueeze(0)
        for i in range(int(self.control_freq*(self.run_time+10))):
            # j+=1
            tic = time.perf_counter()
            lock.acquire()
            button = button_global.copy()
            observed_state = observed_state_global
            # print("=============================")
            # print(observed_state[...,:45])
            # observed_state[8] = 0#TODO set yaw velocity to 0
            flag = flag_global
            stand_flag = stand_flag_global
            lock.release()
            if stand_flag ==1 and init_flag == 1:
                init_flag = 0
            if stand_flag ==1:
                j +=1
                # print(j)
            if button[9]: #button B
                break
            if flag_global == 9: #terminated
                break

            # power_limit = 10#np.clip(int(step/(motor_freq/2)),1,10)#rank up power limit within half a sec
            # position_limit = np.clip(i/(self.control_freq),0,1)*33.5/  \
            #                  np.squeeze(self.env.Kp)*1.0#np.clip(step/(motor_freq/2),0,1)*23.7/40
            # self.env.setSafetyLimit(power_limit=power_limit,position_limit=position_limit)
            
            # 在这里修改actor 
            state = observed_state
            # action = self.agent.action(state)
            # with torch.no_grad:

            # print(self.env.GetFootContacts())
            obs = self.env.getObservation_isaacsim()
            his = obs[:,45:]
            
            cur_obs = obs[:,-45:]

            latent = self.estimator(his)
            # print("latent shape================0",latent)
            obs = torch.cat((cur_obs,latent),dim=1)
    
            action = self.actor(obs).detach()
            blend = np.clip( j /(self.control_freq*3),0,1)#blend within half a sec
            action = blend*action+(1.0-blend)*(init_motor_angle - self.env.default_pos)
            self.env.actions = action
            #print("obs:",obs[0, -40:-28].cpu().numpy())
            if stand_flag==1:
                # print("policy:",action*0.25)
                # print("obs:",obs[0,9:21])
                # print("tau:",self.env._motor_torques)
                tau = self.env._motor_torques
                
                lllen+=1
                #actions_his = np.vstack((actions_his, (action*0.25).cpu().numpy()))
                dof_pos_his = np.vstack((dof_pos_his, obs[0, 9:21].cpu().detach().numpy()))
                
                dof_vel_his = np.vstack((dof_vel_his, (obs[0, 21:33]*20).cpu().detach().numpy()))
                #print(dof_vel_his,"==")
                tau_his = np.vstack((tau_his,tau))
                #print(np.max(dof_vel_his))
                # print("obs:",obs[0,-30])

                if len(tau_his)%10 == 0:
                    print(len(tau_his))
                '''

                if len(tau_his) == 500:
                # np.save("actions_record",actions_history)
                # actions_history_df = pd.DataFrame(actions_history)
                # actions_history_df.to_excel(path_actions)

                    actions_his_df  = pd.DataFrame(actions_his)
                    actions_his_df.to_excel(f"/home/zby/fall_recover/go1_deployment_2/excle/{file_id}/{time_id}_pol_yes.xlsx")
                    dof_vel_his_df  = pd.DataFrame(dof_vel_his)
                    dof_vel_his_df.to_excel(f"/home/zby/fall_recover/go1_deployment_2/excle/{file_id}/{time_id}_obs_vel.xlsx")
                    dof_pos_his_df  = pd.DataFrame(dof_pos_his)
                    dof_pos_his_df.to_excel(f"/home/zby/fall_recover/go1_deployment_2/excle/{file_id}/{time_id}_obs_pos.xlsx")
                    tau_his_df  = pd.DataFrame(tau_his)
                    tau_his_df.to_excel(f"/home/zby/fall_recover/go1_deployment_2/excle/{file_id}/{time_id}_tau.xlsx")
                '''

            '''
            # action = obs[0,-36:-24]
            
            # actions = torch.zeros_like(action,device="cuda:0")
            # angle = np.sin(i*0.02*np.pi)*30/180*np.pi
            # angle = np.sin(j)*30/180*np.pi
            # index = int(i*0.01)%12
            # print(index)
            # actions[6] = angle
            # if stand_flag==1:
            #     print("policy:",actions)
            #     print("obs:",obs[0,-30])
            #     record_angle["real"].append(obs[0,-30].cpu().item())
            #     record_angle["angle"].append(angle)
            # if  j > 5 and stand_flag:
            #     record = pd.DataFrame.from_dict(record_angle)
            #     # record['record_angle'] = record['record_angle'].str.encode('utf-8')
            #     # with pd.ExcelWriter("./angle_record.xlsx",'auto') as xlsx:
            #     record.to_csv("./angle_record.xlsx",encoding='utf_8')
            #         # print("ok!=============================")
            #     break
                

            # action = torch.tensor((angle),device="cuda:0")  #self.env.nominal_motor_angles+
            # print(action,"actor")



            # action = np.array([action]) if len(np.shape(action)) == 0 else np.array(action)
            # action = np.tanh(np.arctanh(action) + np.array([0.3, -0.2, 0, -0.3, -0.2, 0, 0.3, 0.00, 0, -0.3, 0.00, 0]))#option1
            # action = np.clip(action, self.config.conf['actor-output-bounds'][0], self.config.conf['actor-output-bounds'][1])

            # control_action = np.array(action)
            # control_action = rescale(control_action, self.config.conf['actor-output-bounds'],
            #                          self.config.conf['action-bounds'])
            # control_action = np.clip(control_action, self.config.conf['action-bounds'][0],
            #                          self.config.conf['action-bounds'][1])
            lower_bound = torch.tensor([-1.047, -1.047, -1.047, -1.047, -0.663,-0.663,-0.663,-0.663, -2.721, -2.721, -2.721, -2.721],device="cuda:0")
            upper_bound = torch.tensor([1.047, 1.047, 1.047, 1.047, 2.966, 2.966, 2.966, 2.966, 0.837, 0.837, 0.837, 0.837],device="cuda:0")
            # control_action = torch.clip(action, lower_bound, upper_bound)
            # control_action = self.env.limitAction(control_action) #limit action in policy loop
            # print("policy thread command action", control_action)
            # stand up
            # control_action = torch.zeros_like(control_action,device="cuda:0")
'''

            lock.acquire()
            policy_action_global = action#control_action
            flag_global = 2 #policy generarted command
            lock.release()
            # print("control actions:",control_action)
            #logging
            '''
            lock.acquire()
            control_action_list.append(np.squeeze(control_action))
            filtered_action_list.append(filtered_action_log)
            observed_state_list.append(np.squeeze(observed_state))

            base_euler_list.append(base_euler_log)
            base_orn_vel_list.append(base_orn_vel_log)
            base_pos_vel_yaw_list.append(base_pos_vel_yaw_log)
            base_pos_vel_list.append(base_pos_vel_log)
            base_euler_from_quat_list.append(base_euler_from_quat_log)
            base_vel_from_foot_list.append(base_vel_from_foot_log)
            base_gravity_vector_list.append(base_gravity_vector_log)
            wireless_remote_list.append(wireless_remote_log)
            motor_angle_list.append(motor_angle_log)
            motor_velocity_list.append(motor_velocity_log)
            motor_torque_list.append(motor_torque_log)
            base_orn_vel_fusion_list.append(base_orn_vel_fusion_log)
            yaw_vel_from_foot_list.append(yaw_vel_from_foot_log)
            foot_force_list.append(foot_force_log)
            base_imu_list.append(base_imu_log)
            calibrated_acc_list.append(calibrated_acc_log)
            lock.release()
            '''
            toc = time.perf_counter()
            duration = toc-tic
            delay = np.clip(self.control_dt-duration, 0.0, self.control_dt)

            #print("policy duration", duration)

            '''duara_his = np.vstack((duara_his,duration))
            
            if(len(duara_his)==500):
                duara_his_df  = pd.DataFrame(duara_his)
                duara_his_df.to_excel(f"/home/zby/fall_recover/go1_deployment_2/excle/{file_id}/{time_id}_duration.xlsx")
            '''
            # print("policy time delay", delay)
            policy_duration = 0.9*policy_duration+0.1*duration
            time.sleep(delay)

        print("policy avg duration", policy_duration)

        # print("policy duration", duration)

        '''
        filehandler = open(self.dir_path + '/base_orn_vel_base.obj', "wb")
        pickle.dump(base_orn_vel_list, filehandler)
        filehandler = open(self.dir_path + '/base_pos_vel_yaw.obj', "wb")
        pickle.dump(base_pos_vel_yaw_list, filehandler)
        filehandler = open(self.dir_path + '/base_vel_from_foot.obj', "wb")
        pickle.dump(base_vel_from_foot_list, filehandler)
        filehandler = open(self.dir_path + '/base_pos_vel.obj', "wb")
        pickle.dump(base_pos_vel_list, filehandler)
        filehandler = open(self.dir_path + '/base_euler.obj', "wb")
        pickle.dump(base_euler_list, filehandler)
        filehandler = open(self.dir_path + '/base_euler_from_quat.obj', "wb")
        pickle.dump(base_euler_from_quat_list, filehandler)
        filehandler = open(self.dir_path + '/base_gravity_vector.obj', "wb")
        pickle.dump(base_gravity_vector_list, filehandler)
        filehandler = open(self.dir_path + '/control_action.obj', "wb")
        pickle.dump(control_action_list, filehandler)
        filehandler = open(self.dir_path + '/filtered_action.obj', "wb")
        pickle.dump(filtered_action_list, filehandler)
        filehandler = open(self.dir_path + '/observed_state.obj', "wb")
        pickle.dump(observed_state_list, filehandler)
        filehandler = open(self.dir_path + '/wireless_remote.obj', "wb")
        pickle.dump(wireless_remote_list, filehandler)
        filehandler = open(self.dir_path + '/motor_angles.obj', "wb")
        pickle.dump(motor_angle_list, filehandler)
        filehandler = open(self.dir_path + '/motor_velocities.obj', "wb")
        pickle.dump(motor_velocity_list, filehandler)
        filehandler = open(self.dir_path + '/motor_torques.obj', "wb")
        pickle.dump(motor_torque_list, filehandler)
        filehandler = open(self.dir_path + '/yaw_vel_from_foot.obj', "wb")
        pickle.dump(yaw_vel_from_foot_list, filehandler)
        filehandler = open(self.dir_path + '/base_orn_vel_fusion.obj', "wb")
        pickle.dump(base_orn_vel_fusion_list, filehandler)
        filehandler = open(self.dir_path + '/foot_force.obj', "wb")
        pickle.dump(foot_force_list, filehandler)
        filehandler = open(self.dir_path + '/base_imu.obj', "wb")
        pickle.dump(base_imu_list, filehandler)
        filehandler = open(self.dir_path + '/calibrated_acc.obj', "wb")
        pickle.dump(calibrated_acc_list, filehandler)
        '''
        return

    def robot(self, lock):
        global observed_state_global
        global policy_action_global
        global button_global
        global flag_global
        global stand_flag_global
        #log variables
        global control_action_log
        global filtered_action_log
        global observed_state_log

        global base_euler_log
        global base_orn_vel_log
        global base_pos_vel_yaw_log
        global base_pos_vel_log
        global base_euler_from_quat_log
        global base_vel_from_foot_log
        global base_gravity_vector_log
        global wireless_remote_log
        global motor_angle_log
        global motor_velocity_log
        global motor_torque_log
        global base_orn_vel_fusion_log
        global yaw_vel_from_foot_log
        global foot_force_log
        global base_imu_log
        global calibrated_acc_log
        global lllen
        global actions_no_his
        global actions_yes_his
        global cca_his

        ready_flag = 0
        PD_duration = 0
        self.walk_flag = 0
        #initialize sensor readings and policy to prevent abrupt motions at the start
        step = 0
        duration_list = []
        acc_filt = np.zeros(3)
        foot_force_filt = np.zeros(4)
        for i in range(int(self.motor_freq*1)):
            
            # self.env.receiveState()  # wait and fill state
            self.env.receiveObservation()
            obs = self.env.getObservation_isaacsim()
            # self.env.filterObservation()
            # acc_filt = 0.9 * acc_filt + 0.1 * np.squeeze(self.env._raw_state.imu.accelerometer)
            foot_force_filt = 0.9 * foot_force_filt + 0.1 * np.squeeze(self.env._raw_state.footForce)
            self.env.foot_force_offset = foot_force_filt
            # imu_offset = np.array([0, 0, np.linalg.norm(acc_filt)])
            # self.env._velocity_estimator.reset(imu_offset=imu_offset)
            # self.env.filterAction(self.env._motor_angles)#start filtering to prevent sudden changes
            # print("motor angles",self.env._motor_angles)

            #initialize policy
            # observed_state = np.nan_to_num(np.squeeze(self.env.observation_filtered), copy=True, nan=0)#replace nan with 0
            lock.acquire()
            observed_state_global = obs
            lock.release()

            self.env.udp.Send()
            time.sleep(self.motor_dt)

        lock.acquire()
        flag_global = 1#finish initialization
        lock.release()
        
        print("wait for button A")
        while 1: #wait for button A before executing
            lock.acquire()
            flag = flag_global  # finish initialization
            lock.release()
            if self.env.button[8]:
                ready_flag = 1
                print("--------------ready--------------")
            if ready_flag and flag==2: #button A
                break
            self.env.receiveObservation()
            # self.env.getObservation_isaacsim_new()
            # self.env.filterObservation()
            self.env.udp.Send()
            time.sleep(0.05)
        print("start execution")

        init_motor_angle =torch.tensor([self.env._raw_motor_angles[self.env.dof_map_isaacsim[i]] for i in range(12)],device=self.env.model_device,dtype=torch.float32).unsqueeze(0)
        # control_action = np.squeeze(np.array(self.env._motor_angles))
        control_action = torch.zeros((1, 12), device= "cuda:0", dtype= torch.float32)
        
        self.env.stand(5)  #stand up
        # self.env.keep_stand2()
        # return

        
        llen=lllen
        for step in range(int(self.motor_freq*self.run_time)):
            #step = step+1
            # print("step:",step)

            tic = time.perf_counter()
            if self.env.button[9]: #button B
                break
            power_limit = 10  #np.clip(int(step/(motor_freq/2)),1,10)#rank up power limit within half a sec
            position_limit = np.clip(step/(5*self.motor_freq),0,1)*33.5/75  
            # print(position_limit)
                             #np.squeeze(self.env.Kp)*3.0#np.clip(step/(motor_freq/2),0,1)*23.7/40
            position_limit = torch.tensor(position_limit, device="cuda:0")
            self.env.setSafetyLimit(power_limit=power_limit, position_limit=position_limit)#position_limit=position_limit

            self.env.receiveObservation()
            # obs = self.env.getObservation_isaacsim()
            # self.env.filterObservation()

            # print(self.env.filtered_target_vel)

            observed_state = obs  #np.nan_to_num(np.squeeze(self.env.observation_filtered), copy=True, nan=0)#replace nan with 0
            lock.acquire()
            observed_state_global = observed_state
            control_action = policy_action_global
            button_global = self.env.button.copy()
            
            lock.release()

            #blend for smoother action
            blend = np.clip(step/(self.motor_freq/2),0,1)#blend within half a sec
            control_action = blend*control_action+(1.0-blend)*init_motor_angle
            # print("before limit:",control_action)
            control_action = self.env.limitAction(control_action)  # limit action in policy loop
            # print("after limit:",control_action)

            cca=control_action[0]
            cca = torch.tensor(cca,device="cuda:0")

            control_action = np.array(control_action.cpu().detach())

            control_action = self.env.filterAction(control_action)
            # control_action = torch.tensor(control_action,device="cuda:")
            # print("after filter:",control_action)
            control_action = torch.tensor(control_action,device="cuda:0")
            # xsdee43
            stand_flag=stand_flag_global
            if stand_flag==1:
                if llen!=lllen:
                    actions_no_his = np.vstack((actions_no_his, (policy_action_global*0.25).cpu().numpy()))
                    actions_yes_his = np.vstack((actions_yes_his, (control_action*0.25).cpu().numpy()))
                    cca_his=np.vstack((cca_his, (cca*0.25).cpu().numpy()))
                    llen=lllen
                '''
                if len(actions_no_his) == 500:
                    actions_no_his_df  = pd.DataFrame(actions_no_his)
                    actions_no_his_df.to_excel(f"/home/zby/fall_recover/go1_deployment_2/excle/{file_id}/{time_id}_pol_no.xlsx")
                    actions_yes_his_df  = pd.DataFrame(actions_yes_his)
                    actions_yes_his_df.to_excel(f"/home/zby/fall_recover/go1_deployment_2/excle/{file_id}/{time_id}_pol_yes.xlsx")
                    cca_his_df  = pd.DataFrame(cca_his)
                    cca_his_df.to_excel(f"/home/zby/fall_recover/go1_deployment_2/excle/{file_id}/{time_id}_pol_cca.xlsx")
                '''
            
            # print("state", observed_state)
            # self.env.udp.SetSend(self.env.cmd)
            # self.env.udp.Send()
            # position control mode
            # print("np.array([motor.q for motor in self.env.state.motorState[:12]])", np.array([motor.q for motor in self.env.state.motorState[:12]]))
            # print("self.env._motor_angles",self.env._motor_angles)
            # print("np.array([motor.q for motor in self.env._raw_state.motorState[:12]]", np.array([motor.q for motor in self.env._raw_state.motorState[:12]]))
            # print("raw joint-process joint3", np.array([motor.q for motor in self.env.state.motorState[:12]])-self.env._motor_angles)
            # print("raw joint-process joint4",
            #       np.array([motor.q for motor in self.env.state.motorState[:12]]) -
            #       np.array([motor.q for motor in self.env._raw_state.motorState[:12]]))

            lock.acquire()
            flag = flag_global  # finish initialization
            lock.release()

            if flag == 2 and self.env.ry>0.1: # wait for policy to generate valid command
                # print("robot thread command action", self.env.command_action)
                # print("robot thread command action diff", self.env.command_action - self.env._motor_angles)
                # self.env.udp.Send()
                # self.env.applyAction(self.env.command_action, self.env._motor_control_mode)
                lock.acquire()
                stand_flag_global = 1
                lock.release()
                self.env.applyAction( control_action, self.env._motor_control_mode)
                self.walk_flag = 1
            elif flag == 2 and self.walk_flag == 1:
                lock.acquire()
                stand_flag_global = 0
                lock.release()
                # self.env.keep_stand()  
                break
                
            else:
                self.env.keep_stand() 

                self.env.udp.Send()
            # self.env.applyAction(self.env.command_action, self.env._motor_control_mode)

            # #log
            # lock.acquire()
            # #control_action_list.append(np.squeeze(control_action))
            # filtered_action_log = np.squeeze(self.env.command_action).copy()#filtered_action_list.append(np.array(np.squeeze(self.env.command_action)))
            # #observed_state_list.append(np.squeeze(observed_state))

            # base_euler_log = np.squeeze(self.env._base_euler).copy()#base_euler_list.append(np.squeeze(self.env._base_euler))
            # base_orn_vel_log = np.squeeze(self.env.base_orn_vel_base).copy()#base_orn_vel_list.append(np.squeeze(self.env.base_orn_vel_base))
            # base_pos_vel_yaw_log = np.squeeze(self.env.base_pos_vel_yaw).copy()#base_pos_vel_yaw_list.append(np.squeeze(self.env.base_pos_vel_yaw))
            # base_pos_vel_yaw_log = np.squeeze(self.env._velocity_estimator.estimated_velocity).copy()#base_pos_vel_list.append(np.squeeze(self.env._velocity_estimator.estimated_velocity))
            # base_euler_from_quat_log = np.squeeze(self.env.base_euler_from_quat).copy()#base_euler_from_quat_list.append(np.squeeze(self.env.base_euler_from_quat))
            # base_vel_from_foot_log = np.squeeze(self.env._velocity_estimator.base_vel_from_foot).copy()#base_vel_from_foot_list.append(np.squeeze(self.env._velocity_estimator.base_vel_from_foot))
            # base_gravity_vector_log = np.squeeze(self.env.gravityPosInBase).copy()#base_gravity_vector_list.append(np.squeeze(self.env.gravityPosInBase))
            # wireless_remote_log = np.squeeze(self.env.wireless_remote).copy()#wireless_remote_list.append(self.env.wireless_remote)
            # motor_angle_log = np.squeeze(self.env._motor_angles).copy() #motor_angle_list.append(np.squeeze(np.array(self.env._motor_angles)))
            # motor_velocity_log = np.squeeze(self.env._motor_velocities).copy()#motor_velocity_list.append(np.squeeze(np.array(self.env._motor_velocities)))
            # motor_torque_log = np.squeeze(self.env._motor_torques).copy()#motor_torque_list.append(np.squeeze(np.array(self.env._motor_torques)))
            # base_orn_vel_fusion_log = np.squeeze(self.env._base_angular_velocity_fusion).copy()#base_orn_vel_fusion_list.append(self.env._base_angular_velocity_fusion)
            # yaw_vel_from_foot_log = np.squeeze(self.env._velocity_estimator.yaw_vel_from_foot).copy()#yaw_vel_from_foot_list.append(np.squeeze(np.array(self.env._velocity_estimator.yaw_vel_from_foot)))
            # foot_force_log = np.squeeze(self.env._foot_force).copy()#foot_force_list.append(np.array(self.env._foot_force))
            # base_imu_log = np.squeeze(self.env.state.imu.accelerometer).copy()#base_imu_list.append(np.squeeze(np.array(self.env.state.imu.accelerometer)))
            # calibrated_acc_log = np.squeeze(self.env._velocity_estimator.calibrated_acc).copy()#calibrated_acc_list.append(np.squeeze(np.array(self.env._velocity_estimator.calibrated_acc)))
            # lock.release()

            toc = time.perf_counter()
            duration = toc-tic
            delay = np.clip(self.motor_dt-duration,0.0,self.motor_dt)

            # print("PD duration", duration)
            # print("PD time delay", delay)
            PD_duration = 0.9*PD_duration+0.1*duration
            time.sleep(delay)

        print("PD_duration", PD_duration)
        lock.acquire()
        flag_global = 9 #finish initialization
        lock.release()

        #rest robot
        self.env.rest(5)

        return

    def run(self):
        # creating a lock
        lock = threading.Lock()

        # creating threads
        t1 = threading.Thread(target=self.robot, args=(lock,))
        t2 = threading.Thread(target=self.policy, args=(lock,))

        # start threads
        t1.start()
        t2.start()

        # wait until threads finish their job
        t1.join()
        t2.join()

        return



def main(): 
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    os.sys.path.insert(0, parentdir)
    # 
    test = Run()
    test.run()

    # a = test.get_actor3()
    # x = torch.ones((1,270),device="cuda:0")
    # print(a(x))
    # control_action = torch.zeros((1,12),device="cuda:0")
    # env = Go1_robot(motor_command_freq=500,control_freq=25)
    # env.udp.Send()
    # global observed_state_global
    # while 1:
    #     obs1 = env.receiveObservation()
    #     obs = env.getObservation_isaacsim()
    #     # observed_state_global = obs.copy()
    #     # print(a(obs))
    #     # print(obs[:,:45])
    #     time.sleep(0.5)
        
    #     control_action_d = env.limitAction(control_action)
    #     print(control_action_d)
    #     print("=============================")

if __name__ == '__main__':
    main()
#    actions_his_df  = pd.DataFrame(actions_his)
#    actions_his_df.to_excel(f"/home/zby/fall_recover/go1_deployment_2/excle/{file_id}/{time_id}_pol_yes.xlsx")
    dof_vel_his_df  = pd.DataFrame(dof_vel_his)
    dof_vel_his_df.to_excel(f"/home/zby/fall_recover/go1_deployment_2/excle/{file_id}/{time_id}_obs_vel.xlsx")
    dof_pos_his_df  = pd.DataFrame(dof_pos_his)
    dof_pos_his_df.to_excel(f"/home/zby/fall_recover/go1_deployment_2/excle/{file_id}/{time_id}_obs_pos.xlsx")
    tau_his_df  = pd.DataFrame(tau_his)
    tau_his_df.to_excel(f"/home/zby/fall_recover/go1_deployment_2/excle/{file_id}/{time_id}_tau.xlsx")
    actions_no_his_df  = pd.DataFrame(actions_no_his)
    actions_no_his_df.to_excel(f"/home/zby/fall_recover/go1_deployment_2/excle/{file_id}/{time_id}_pol_no.xlsx")
    actions_yes_his_df  = pd.DataFrame(actions_yes_his)
    actions_yes_his_df.to_excel(f"/home/zby/fall_recover/go1_deployment_2/excle/{file_id}/{time_id}_pol_yes.xlsx")
    cca_his_df  = pd.DataFrame(cca_his)
    cca_his_df.to_excel(f"/home/zby/fall_recover/go1_deployment_2/excle/{file_id}/{time_id}_pol_cca.xlsx")