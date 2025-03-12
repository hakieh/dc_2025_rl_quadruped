import torch


import torch
import torch.nn as nn
import torch.nn.functional as F

class Estimator(nn.Module): 
    def __init__(self):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Linear(245,512),
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
            nn.Linear(59,1024),
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


loaded = torch.load("./weights/shake_con/model_5750.pt")
# print(1)
actor = alg()
estimator = Estimator()

actor.load_state_dict(loaded['model_state_dict'],strict=False)
actor.to(torch.device("cuda"))

estimator.load_state_dict(loaded['estimator_state_dict'])
estimator.to(torch.device("cuda"))

a = torch.ones((1,245),device="cuda:0")
print(estimator(a))
b = torch.ones((1,59),device="cuda:0")
print(actor(b))


