from ctypes import pointer
from click import pass_context
import pybullet as p
import pybullet_envs
import pybullet_data
import torch 
import gym
from gym import spaces
import time
from stable_baselines3 import PPO 
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
import numpy as np        
import math
import os
import inv_kine.inv_kine as ik
from stable_baselines3.common.callbacks import BaseCallback
from dog_env import TestudogEnv
file_id=time.strftime("%Y-%m-%d",time.localtime())
time_id=time.strftime("%H:%M",time.localtime())
# see tensorboard : tensorboard --logdir=log (open terminal in final_project dir)
class CustomCheckpointCallback(BaseCallback):

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = 'model', verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq          
        self.save_path = save_path          
        self.name_prefix = name_prefix     
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"{time_id}_{self.name_prefix}_{self.num_timesteps}.pt")
            state={}
            mlp_state=self.model.policy.mlp_extractor.state_dict()
            for i in range(0,5,2):
                state[f'actor.{i}.weight']=mlp_state[f'policy_net.{i}.weight']
                state[f'actor.{i}.bias']=mlp_state[f'policy_net.{i}.bias']
                state[f'critic.{i}.weight']=mlp_state[f'value_net.{i}.weight']
                state[f'critic.{i}.bias']=mlp_state[f'value_net.{i}.bias']
            state['actor.6.weight']=self.model.policy.action_net.state_dict()['weight']
            state['actor.6.bias']=self.model.policy.action_net.state_dict()['bias']
            state['critic.6.weight']=self.model.policy.value_net.state_dict()['weight']
            state['critic.6.bias']=self.model.policy.value_net.state_dict()['bias']
            torch.save({
                'model_state_dict': {
                    **state
                },
                'optimizer_state_dict':{
                    **self.model.policy.optimizer.state_dict()
                }
            }, model_path)
            
            if self.verbose > 0:
                print(f"模型已保存: {model_path}")
        
        return True

def load_model(mm:PPO):
    weights = torch.load("./weights/train/2025-03-04/20:31_model_2000000.pt")
    state_mlp={}
    state_pol={}
    state_val={}
    for i in range(0,5,2):
        state_mlp[f'policy_net.{i}.weight']=weights['model_state_dict'][f'actor.{i}.weight']
        state_mlp[f'policy_net.{i}.bias']=weights['model_state_dict'][f'actor.{i}.bias']
        state_mlp[f'value_net.{i}.weight']=weights['model_state_dict'][f'critic.{i}.weight']
        state_mlp[f'value_net.{i}.bias']=weights['model_state_dict'][f'critic.{i}.bias']
    state_pol['weight']=weights['model_state_dict']['actor.6.weight']
    state_pol['bias']=weights['model_state_dict']['actor.6.bias']
    state_val['weight']=weights['model_state_dict']['critic.6.weight']
    state_val['bias']=weights['model_state_dict']['critic.6.bias']

    mm.policy.mlp_extractor.load_state_dict(state_mlp,strict=False)
    mm.policy.action_net.load_state_dict(state_pol,strict=False)
    mm.policy.value_net.load_state_dict(state_val,strict=False)

    return mm

if (__name__ == '__main__'):

    
    env = TestudogEnv()
    # check_env(env)
    policy_kwargs = dict(activation_fn=torch.nn.ELU,
                    net_arch=dict(pi=[128,128,128], vf=[128,128,128]))
    model = PPO(
    "MlpPolicy",
    env,
    
    # learning_rate=0.0005766503906250003,         
    verbose=2,
    tensorboard_log='./my_log',
    policy_kwargs=policy_kwargs,
    device='cpu',
    # target_kl=0.5
    gamma=0.95,
    learning_rate=3e-4,
    n_steps=4096, 
    batch_size=256
    )
    checkpoint_callback = CustomCheckpointCallback(save_freq=50000, save_path=f'./weights/train/{file_id}', name_prefix='model')

    # model=load_model(model)
    model.learn(total_timesteps=6000000, callback=checkpoint_callback)

    model.save("./weights/train/aa.zip")
    env.close()
    
    # train loop   
    # while(True):
    #     print(count)
    #     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    #     model.save(f"{model_dir}/{TIMESTEPS*count}")
    #     count += 1
    #     if True == False:
    #         break
    