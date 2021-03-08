import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import os


import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.nn.utils import clip_grad_norm_
from unityagents import UnityEnvironment
import numpy as np
from collections import deque, namedtuple
import random
import copy
import matplotlib.pyplot as plt

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class ActorNetwork(nn.Module):
    """ Critic (value) Model. """
    def __init__(self, input_dim, output_dim, fc1_dims=400, fc2_dims=300, seed=42, name="Critic", chkpt_dir="save"):
        """
        Initialize parameters and build model
        
        Param
        =====
            input_dims (int): Input dimension, state size
            n_actions (int): Action space
            fc1_dims (int): Number of nodes in first hidden layer
            fc2_dims (int): Number of nodes in second hidden layer
            name (String): Name of model
            chkpt_dir (String): Directory where checkpoint weights will be store
            
        Return
        ====
            None
        """
        super(ActorNetwork, self).__init__()


        self.seed = torch.manual_seed(seed)
        self.checkpoint_file = os.path.join(chkpt_dir, name+"_ddpg.pth")
        
        self.fc1 = nn.Linear(input_dim,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.fc3 = nn.Linear(fc2_dims,output_dim)
        self.nonlin = f.relu 
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.bn1 = nn.BatchNorm1d(fc1_dims)
        self.bn2 = nn.BatchNorm1d(fc2_dims)
        
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)        
        
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)
                    
        f3 = 3e-3
        self.fc3.weight.data.uniform_(-f3, f3)
        self.fc3.bias.data.uniform_(-f3, f3)     


    def forward2(self, x):
        h1 = self.activation(self.batch_norm_input(self.fc1(x)))
        h2 = self.activation(self.batch_norm_hidden1(self.fc2(h1)))
        h4 = self.fc4(h2)
        return f.tanh(h4)        

    def forward(self, x):
        # return a vector of the force
        h1 = self.nonlin(self.bn1(self.fc1(self.bn0(x))))

        h2 = self.nonlin(self.bn2(self.fc2(h1)))
        h3 = (self.fc3(h2))
        norm = torch.norm(h3)

        return f.tanh(h3)
    
    def save_checkpoint(self):
        """ Save weight's checkpoint"""
        print("... Saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        """ Load weight's checkpoint"""
        print("... Loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))
        
class CriticNetwork(nn.Module):
    """ Critic (value) Model. """
        
    def __init__(self, input_dims , n_actions, fc1_dims=400, fc2_dims=300, name="Critic", chkpt_dir="save"):
        """
        Initialize parameters and build model
        
        Param
        =====
            input_dims (int): Input dimension, state size
            n_actions (int): Action space
            fc1_dims (int): Number of nodes in first hidden layer
            fc2_dims (int): Number of nodes in second hidden layer
            name (String): Name of model
            chkpt_dir (String): Directory where checkpoint weights will be store
            
        Return
        ====
            None
        """
        
        super(CriticNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_file = os.path.join(chkpt_dir, name+"_ddpg.pth")
        
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims+n_actions, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 1)
        self.relu = nn.ReLU()
        
        self.bn0 = nn.LayerNorm(input_dims)
        self.bn1 = nn.LayerNorm(fc1_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)
                
        self.device = T.device('cuda:0') if T.cuda.is_available() else T.device("cpu")
    
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)
        
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)
        
        f3 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc3.weight.data.uniform_(-f3, f3)
        self.fc3.bias.data.uniform_(-f3, f3)
    
    def forward(self, states, actions):
        """ Build a critic (value) network for mapping (state, action) in Q-values """
        out = self.fc1(self.bn0(states))
        out = self.relu(self.bn1(out))
        out = self.bn2(self.fc2(T.cat([out,actions],1)))
        out = self.relu(out)
        out = self.fc3(out)
        return out
    
    def save_checkpoint(self):
        """ Save weight's checkpoint"""
        print("... Saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        """ Load weight's checkpoint"""
        print("... Loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))
 