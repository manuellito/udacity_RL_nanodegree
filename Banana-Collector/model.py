import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """ Network policy"""
    
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """ Ininitialize paramters and build model.py
        Params
        ======
           state_size (int): Dimension of each state
           action_size (int): Dimension of each action
           seed (int): Random seed
           fc1_units (int): Number of nodes in first hidden layer
           fc2_units (int): Number of nodes in second hidden layer
           
        Return
        ======
        None
        """
        
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
    def forward(self, state):
        """ Predict actions probabilities from state 
        Params
        ======
        state (array): State of each dimension
        
        Return
        actions_probabilities (array): Probabilities for each action
        """
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class QNetworkCNN(nn.Module):
    """ Network Policy for CNN """
    
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """ Ininitialize paramters and build model.py
        Params
        ======
           state_size (int): Dimension of each state
           action_size (int): Dimension of each action
           seed (int): Random seed
           fc1_units (int): Number of nodes in first hidden layer
           fc2_units (int): Number of nodes in second hidden layer
           
        Return
        ======
        None
        """
        
        super(QNetworkCNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_pool = nn.MaxPool2d(3,3)        
        
        self.fc1 = nn.Linear(14*14*64, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        print(self)
        
    def forward(self, state):
        """ Predict actions probabilities from state 
        Params
        ======
        state (array): State of each dimension
        
        Return
        actions_probabilities (array): Probabilities for each action
        """
        
        x = self.conv1(state)
        x = self.conv1_pool(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_pool(x)
        x = F.relu(x)
        x = x.reshape(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)        