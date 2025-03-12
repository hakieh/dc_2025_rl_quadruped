import gymnasium as gym
from quad_env import QuadEnv
from actorcritic import CustomActorCriticPolicy,CustomCheckpointCallback
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import torch
import time
file_id=time.strftime("%Y-%m-%d",time.localtime())
time_id=time.strftime("%Y-%m-%d",time.localtime())

if __name__ == '__main__':
    env=QuadEnv()
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])  # 用 DummyVecEnv 包装环境

    # 检查环境是否兼容
    check_env(env)

    model = SAC(
    CustomActorCriticPolicy, 
    env, 
    # action_noise=action_noise,
    learning_rate=0.0005766503906250003,
    buffer_size=1000000,  # 经验回放缓冲区大小
    batch_size=256,
    tau=0.005,  # 目标网络软更新系数
    gamma=0.99,  # 折扣因子
    train_freq=1,  # 每个步骤都训练
    gradient_steps=1,  # 每次更新的梯度步数
    verbose=1,
    device="cuda" if torch.cuda.is_available() else "cpu"
)


    weights = torch.load("./weights/train/model_3500.pt")

    model.policy.actor.load_state_dict(weights['model_state_dict'],strict=False)
    model.policy.critic.load_state_dict(weights['model_state_dict'],strict=False)
    model.policy.optimizer.param_groups[0]['eps'] = 1e-08
    model.policy.optimizer.state=weights['optimizer_state_dict']['state']

    checkpoint_callback = CustomCheckpointCallback(save_freq=20000, save_path=f'./weights/train/{file_id}', name_prefix='model')


    model.learn(total_timesteps=200000, callback=checkpoint_callback)

    # model.save("./weights/train/aa.pt")
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