# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
args_cli.task = "Isaac-Velocity-Flat-Unitree-Go1-Play-v0"
args_cli.num_envs = 1
# args_cli.load_run = "2024-11-20_17-06-23"
# args_cli.checkpoint = "model_750"

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    # log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    # log_root_path = os.path.abspath(log_root_path)
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.join("/data/hkh/WEIGHT/bipedal", log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # reset environment
    obs, infos = env.get_observations()
    critic_obs = infos["observations"].get("critic", obs)
    obs_history = infos["observations"].get("history", obs)
    timestep = 0
    t  = torch.tensor(0)
    # simulate environment
    time_flag = datetime.now().strftime("%Y-%m-%d-%H-%M")
    path_actions = "/data/hkh/excle/real_sim/action_record_"+time_flag+".xlsx"
    path_obs = "/data/hkh/excle/real_sim/dof_pos_"+time_flag+".xlsx"
    a = {"angle":[],"real":[]}
    actions_history = np.zeros([1,12])
    dof_pos_his = np.zeros([1,12])
    action_record = np.load("/data/hkh/code/IsaacLab_bipedal/source/standalone/workflows/rsl_rl/actions_record.npy")
    action_record = torch.from_numpy(action_record).to("cuda:0")
    i = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # print(obs)
            # obs = torch.zeros_like(obs,device="cuda:0")
            # obs_history = torch.zeros_like(obs_history)
            # obs = torch.zeros_like(obs)
            # actions = policy(obs,obs_history=obs_history)
            actions = action_record[i,:].unsqueeze(0)
            i+=1
            print("-------------------------")
            print("current:",obs[0,9:21])
            
            # actions = 4*obs[:,9:21]
            # actions = torch.ones_like(actions,device="cuda:0")
            print("actions:",actions[0,:]*0.25)
           
            
            # print(obs)
            # print("==")
            # print(obs[0,9:21])
            # print(actions)
            # actions = torch.zeros_like(actions)
            # actions[:,8] = 2
            # t+=1
            
            # env stepping
            obs, _, _, infos = env.step(actions)
            actions_history = np.vstack((actions_history, actions.cpu().numpy()))
            dof_pos_his = np.vstack((dof_pos_his, obs[0,9:21].cpu().numpy()))
            obs_history = infos["observations"].get("history", obs)
            obs_history = obs_history.to("cuda:0")
            # break
            if len(actions_history) == 999:
                # np.save("actions_record",actions_history)
                # actions_history_df = pd.DataFrame(actions_history)
                # actions_history_df.to_excel(path_actions)

                dof_pos_his_df = pd.DataFrame(dof_pos_his)
                dof_pos_his_df.to_excel(path_obs)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
