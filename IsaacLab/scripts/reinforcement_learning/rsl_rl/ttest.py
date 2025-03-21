import torch
print(torch.cuda.is_available())  # 应该返回 True
# print(torch.version.cuda)        # 应该输出 CUDA 版本号
'''./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py  --task Isaac-Velocity-Rough-Unitree-Go1-v0 --num_envs 4096 --headless --max_iterations 100000
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py  --task Isaac-Velocity-Flat-Unitree-Go1-v0 --num_envs 1 
'''











