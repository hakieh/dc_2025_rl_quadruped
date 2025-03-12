import torch
import numpy as np
import random
a=[(random.random()*2-1),(random.random()*2-1),0]
print(torch.tensor(a).unsqueeze(0))