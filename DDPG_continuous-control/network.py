import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import os

class ActorNetwork(nn.Module):
    """ Actor (policy) Model. """
    
    def __init__(self, lr_actor, input_dims, n_actions, fc1_dims=400, fc2_dims=300, name="Actor", chkpt_dir="save"):
        """
        Initialize parameters and build model
        
        Param
        =====
            lr_actor (float): Learning rate for Actor optimizer
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
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_file = os.path.join(chkpt_dir, name+"_ddpg.pth")
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn0 = nn.LayerNorm(self.input_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)
        
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)
                    
        f3 = 3e-3
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)
            
        self.optimizer = optim.Adam(self.parameters(), lr=lr_actor)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)
            
    def forward(self, state):
        """ Build an actor (policy) network for mapping states in actions """
        x = self.fc1(self.bn0(state))
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))
        
        return x
        
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
        
    def __init__(self, lr_critic, input_dims , n_actions, fc1_dims=400, fc2_dims=300, name="Critic", chkpt_dir="save"):
        """
        Initialize parameters and build model
        
        Param
        =====
            lr_critic (float): Learning rate for Critic optimizer
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
                
        self.optimizer = optim.Adam(self.parameters(), lr=lr_critic)
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
 