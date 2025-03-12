import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
file_id=time.strftime("%Y-%m-%d",time.localtime())
time_id=time.strftime("%H:%M",time.localtime())
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

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ELU(),
            nn.Linear(128, features_dim),
            nn.ELU(),
        )

    def forward(self, observations):
        return self.net(observations)

class Actor(nn.Module):
    def __init__(self, features_dim=45):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 12)
        )

    def forward(self, x):
        return self.actor(x)

class Critic(nn.Module):
    def __init__(self, features_dim=45):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.critic(x)
