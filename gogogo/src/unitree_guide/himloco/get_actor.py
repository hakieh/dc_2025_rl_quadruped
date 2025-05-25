
from collections import OrderedDict
import json
import torch
import sys
sys.path.append("/home/zby/fall_recover/go1_deployment_2/himloco")
# from .modules import him_actor_critic, him_estimator
from modules.him_actor_critic import HIMActorCritic



def get_actor():
    with open("/home/zby/fall_recover/go1_deployment_2/himloco/modules/config.json", "r") as f:
        config_dict = json.load(f, object_pairs_hook= OrderedDict)
    num_obs = 270
    num_critic_obs = 238
    num_one_step_obs = 45
    num_actions = 12
    policy_cfg = config_dict["policy_loco"]
    actor_critic: HIMActorCritic = HIMActorCritic(  num_obs,
                                                    num_critic_obs,
                                                    num_one_step_obs,
                                                    num_actions,
                                                    **policy_cfg).to("cpu")
    
    path = "/home/zby/fall_recover/go1_deployment_2/himloco/weight/model_23280.pt"
    loaded  = torch.load(path,map_location="cpu")
    actor_critic.load_state_dict(loaded['model_state_dict'])
    actor_critic.to(torch.device("cuda"))
    actor = actor_critic.act_inference
    print("------------loading action model successfully----------------")
    return actor





if __name__ == "__main__":

    # with open("./modules/config.json", "r") as f:
    #     config_dict = json.load(f, object_pairs_hook= OrderedDict)
    get_actor()
