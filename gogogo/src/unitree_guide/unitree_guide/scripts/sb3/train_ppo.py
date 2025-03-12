import gymnasium as gym
from quad_env import QuadEnv
from actorcritic import CustomCheckpointCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import torch
import time
file_id=time.strftime("%Y-%m-%d",time.localtime())
time_id=time.strftime("%H:%M",time.localtime())

def load_model(mm:PPO):
    weights = torch.load("./weights/train/2025-03-06/15:27_model_150000.pt")
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

if __name__ == '__main__':
    env = QuadEnv()
    # env = make_vec_env(lambda: env, n_envs=4)
    check_env(env)
    policy_kwargs = dict(activation_fn=torch.nn.ELU,
                    net_arch=dict(pi=[128,128,128], vf=[128,128,128]))
    model = PPO(
    "MlpPolicy",
    env,
    # learning_rate=0.0005766503906250003,         
    verbose=2,
    tensorboard_log='./log',
    policy_kwargs=policy_kwargs,
    device='cpu',
    gamma=0.95,
    learning_rate=3e-4,
    n_steps=4096, 
    batch_size=256
    )

    # model=load_model(model)
    print(model.policy.mlp_extractor.policy_net)
    print(model.policy.mlp_extractor.value_net)
    print(model.policy.features_extractor_class)
    # print(state)
    # model.policy.actor.load_state_dict(weights['model_state_dict'],strict=False)
    # model.policy.critic.load_state_dict(weights['model_state_dict'],strict=False)
    # model.policy.optimizer.state=weights['optimizer_state_dict']['state']
    # model.policy.optimizer.param_groups[0]['eps'] = 1e-08
    # model.policy.log_std.data.fill_(1.0)
    

    checkpoint_callback = CustomCheckpointCallback(save_freq=50000, save_path=f'./weights/train/{file_id}', name_prefix='model')


    model.learn(total_timesteps=6000000, callback=checkpoint_callback)

    model.save("./weights/train/aa.zip")
    env.close()
    '''
---------------------------------------
| rollout/                |           |
|    ep_len_mean          | 1.71e+03  |
|    ep_rew_mean          | 446       |
| time/                   |           |
|    fps                  | 24        |
|    iterations           | 11        |
|    time_elapsed         | 922       |
|    total_timesteps      | 22528     |
| train/                  |           |
|    approx_kl            | 6.5268965 |
|    clip_fraction        | 0.996     |
|    clip_range           | 0.1       |
|    entropy_loss         | -17.7     |
|    explained_variance   | 0.00251   |
|    learning_rate        | 0.0001    |
|    loss                 | 0.769     |
|    n_updates            | 100       |
|    policy_gradient_loss | 0.373     |
|    std                  | 1.07      |
|    value_loss           | 0.662     |
---------------------------------------
'''
'''
tensorboard --logdir ./log/PPO_1 --port 6006
'''