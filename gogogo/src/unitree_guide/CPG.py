import torch, torch.nn as nn

import numpy as np
# from collections import Iterable, Iterator
from typing import Optional, Dict, List, Union, Tuple, TypeVar

from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer
from torch import Tensor
from scipy.signal import find_peaks
from matplotlib import pyplot as plt


class ODEBase(nn.Module):
    '''Base class for all modules have kinetic equations。'''
    def __init__(self) -> None:
        super().__init__()
        pass
    
    def derivative(self):
        '''Equation derivated'''
        raise NotImplementedError()

    def step(self):
        '''Derivated step'''
        raise NotImplementedError()

    def update(self):
        '''Equationn state update'''
        raise NotImplementedError()

    def record(self):
        '''Record trace of state, optional.'''
        raise NotImplementedError()

class MCPG(ODEBase):
    def __init__(self,
                 neuron_nums: int = 18,
                 connect_matrix: Union[float, List[float]] = None,
                 weight: Union[float, List[float]] = None,
                 bound = None,
                 I: Union[float, List[float]] = None,
                 Up: Union[float, List[float]] = None,
                 
                 Vp: Union[float, List[float]] = None,
                               
                 Ta: Union[float, List[float]] = None,
                 Tr: Union[float, List[float]] = None,
                 beta: Union[float, List[float]] = None,
                 t: Union[float, int] = 0,
                 dt: Union[float, int] = 0.003,
                 scale: float = 0.02,
                 device="cuda", dtype=None,
                 learnning: bool = True,
                 #? ds params
                 *arg, **kwargs) -> None:
        super().__init__()
        factory_kwargs = {'device':device, 'dtype':dtype}
        self.neuron_nums = neuron_nums
        #parameters
        self.weight = Parameter(torch.tensor(weight, **factory_kwargs)*torch.ones((neuron_nums,neuron_nums), **factory_kwargs) if weight is not None else  0.5*torch.ones((neuron_nums, neuron_nums), **factory_kwargs))
        self.Ta = Parameter(torch.tensor(Ta, **factory_kwargs)*torch.ones(neuron_nums, **factory_kwargs) if Ta is not None else 1/0.8*torch.ones(neuron_nums, **factory_kwargs))
        self.Tr = Parameter(torch.tensor(Tr, **factory_kwargs)*torch.ones(neuron_nums, **factory_kwargs) if Tr is not None else 1/0.08*torch.ones(neuron_nums, **factory_kwargs))
        self.beta = Parameter(torch.tensor(beta, **factory_kwargs)*torch.ones(neuron_nums, **factory_kwargs) if beta is not None else 3*torch.ones(neuron_nums, **factory_kwargs))
        self.connect_matrix = torch.tensor(connect_matrix, **factory_kwargs) if connect_matrix is not None else (torch.ones((neuron_nums, neuron_nums), **factory_kwargs)-torch.eye(neuron_nums, **factory_kwargs))
        self.bound = bound if bound is not None else 0.4
        #input
        self.I = torch.tensor(I, **factory_kwargs)*torch.ones(neuron_nums, **factory_kwargs) if I is not None else 1*torch.ones(neuron_nums, **factory_kwargs)
        #variables(p means previous while n means now)
        self.Up = torch.tensor(Up, **factory_kwargs)*torch.ones(neuron_nums, **factory_kwargs) if Up is not None else (torch.rand(neuron_nums, **factory_kwargs))/10
        self.Un = self.Up
        self.Vp = torch.tensor(Vp, **factory_kwargs)*torch.ones(neuron_nums, **factory_kwargs) if Vp is not None else (torch.rand(neuron_nums, **factory_kwargs))/50
        self.Vn = self.Vp
        self.fp = (torch.abs(self.Up) + self.Up)/2
        self.fn = 0* torch.ones(neuron_nums,**factory_kwargs)
        #record of all previous variables
        self.fs = 0* torch.ones((1,neuron_nums),**factory_kwargs)
        self.us = torch.reshape(self.Up,(1,neuron_nums))
        self.vs = torch.reshape(self.Vp,(1,neuron_nums))
        #reward
        self.reward = 0
        self.total_reward = 0
        #time
        self.t = t
        self.dt = dt
        self.ts = np.array([t])
        self.scale = scale
        self.learnning = learnning

        for p in self.parameters():
            p.requires_grad = False
    
    def derivative(self,state,inputs):
        #equatation:
        # du = (-u + weight*f - beta*v + input) * Tr
        # dv = (-v + f) * Ta
        # f = abs(u)    
        [U,V] = state
        f = ((torch.abs(U)) + U)/2
        I = inputs
        if(str(f.size())=='torch.Size([1, 12])'):
            f=f[0]
        DU =  (-U - torch.mv(self.connect_matrix*self.weight , f) - self.beta*V + I) *self.Tr
        DV = ( -V + f )* self.Ta
        
        
        return DU, DV
       

    def step(self, I=1):
        dt = self.dt
        self.t += dt
        np.append(self.ts, self.t)
        self.I = I
        #rk4 integrate
        [DU1,DV1]=self.derivative([self.Up,self.Vp],self.I)
        [DU2,DV2]=self.derivative([self.Up+DU1*self.dt/2,self.Vp+DV1*self.dt/2],self.I)
        [DU3,DV3]=self.derivative([self.Up+DU2*self.dt/2,self.Vp+DV2*self.dt/2],self.I)  
        [DU4,DV4]=self.derivative([self.Up+DU3*self.dt,self.Vp+DV3*self.dt],self.I)
        self.Un=self.Up+self.dt*(DU1+2*DU2+2*DU3+DU4)/6
        self.Vn=self.Vp+self.dt*(DV1+2*DV2+2*DV3+DV4)/6
        ###
        #bound
        self.Un[self.Un>self.bound]=self.bound
        self.Un[self.Un<-self.bound]=-self.bound
        self.Vn[self.Vn>self.bound]=self.bound
        self.Vn[self.Vn<-self.bound]=-self.bound
        ### 
        self.fn = (abs(self.Un) + self.Un)/2
        self.fs = torch.cat((self.fs, torch.reshape(self.fn,(1,self.neuron_nums))),dim=0)
        self.us = torch.cat((self.us, torch.reshape(self.Un,(1,self.neuron_nums))),dim=0)
        self.vs = torch.cat((self.vs, torch.reshape(self.Vn,(1,self.neuron_nums))),dim=0)
        #print(torch.stack([self.Un,self.Vn]))    
        return self.Un
    
    def get_output(self):
        return self.fn.numpy() 
    
    def forward(self, x=0, inputs2cpg=None, dt=None):
        x1=x+1
        self.update()
        return self.step(x1)
    
    def get_reward(self,reward):
        self.reward = reward
        self.total_reward = self.total_reward + reward
        
    def count_period(self, target_period = 0, start_place = 500):
        #reward based on periods
        #the value of this function represents the difference between the period and the target period
        signal = self.fs.numpy()
        T, neuron_nums = signal.shape
        periods = []
        for i in range(neuron_nums):
            neuron_signal = np.reshape(signal[start_place:, i], T-start_place)          
            peaks, _ = find_peaks(neuron_signal)
            if len(peaks) < 2:
                return None                         
            peak_times = np.diff(peaks)
            period = np.mean(peak_times) * self.dt  
                   
            if period is not None:
                periods.append(period)
    
        if not periods:
            return 0.0, 0.0
    
        mean_period = np.mean(periods)   
        period_error = (mean_period - target_period)     
        return period_error
    
    def update(self):
        
        self.Up, self.Vp = self.Un, self.Vn
    
    def draw_output(self,neuron=None, title=None):
        #draw the output(f) of neurons 
        neurons = neuron if neuron is not None else range(self.neuron_nums)
        alloutput = self.us. numpy()[:,neurons]
        plt.figure()
        plt.plot(alloutput,label=neurons)
        if title is not None:
            plt.title(title)
        plt.legend()
    def draw_relative_place(self,neuron_num=0):
        #draw the u and v of one neuron
        x = self.us.numpy()[:,neuron_num]
        y = self.vs.numpy()[:,neuron_num]
        plt.figure()
        plt.title('relative_place')
        plt.plot(x,y)
 
if __name__=="__main__":

    connectmatrix = np.array([[0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

    net = MCPG(12,connectmatrix, Ta=1/0.8, Tr=1/0.08, beta=3)
    for i in range(5000):
        net.forward()
        print(net.get_output())

    net.draw_output(neuron=[0,1,2])
    net.draw_output(neuron=[0,3,6,9])
    net.draw_relative_place()
    print(net.count_period(start_place=1000))