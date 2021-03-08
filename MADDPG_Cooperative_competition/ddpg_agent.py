from network import ActorNetwork, CriticNetwork
import torch as torch
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


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

from utils import ReplayBuffer, OUActionNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, input_size, output_size, hidden = 256, lr_actor=1.0e-3, lr_critic=1.0e-3, agent_number=0, tau=1.0e-2,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.99, weight_decay=0, clipgrad=.1, seed = 42):
        super(Agent, self).__init__()
        
        self.seed = seed
        self.actor         = ActorNetwork(input_size, output_size, name=f"Actor_Agent{agent_number}").to(device)
        self.critic        = CriticNetwork(input_size, output_size, name=f"Critic_Agent{agent_number}").to(device)
        self.target_actor  = ActorNetwork(input_size, output_size, name=f"Actor_Target_Agent{agent_number}").to(device)
        self.target_critic = CriticNetwork(input_size, output_size, name=f"Critic_Target_Agent{agent_number}").to(device)
        
        
        
        self.noise = OUActionNoise(mu=np.zeros(output_size))
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_decay=epsilon_decay
        self.gamma = gamma
        self.clipgrad = clipgrad
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)
       

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device) #.unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().squeeze(0).data.numpy()

        self.actor.train()
        if add_noise:
            action += self.noise.sample() * self.epsilon
        return np.clip(action, -1, 1)
    
    
    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        

        

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.target_actor(next_states.to(device))
        #set_trace()
        Q_targets_next = self.target_critic(next_states.to(device), actions_next.to(device))
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic(states, actions)
        critic_loss = f.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), self.clipgrad)
        self.critic_optimizer.step()

        #    update actor
        # Compute actor loss
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        #clip_grad_norm_(self.actor.parameters(), self.clipgrad)
        self.actor_optimizer.step()

        #    update target networks
        self.soft_update(self.critic, self.target_critic )
        self.soft_update(self.actor, self.target_actor)                     
        
        #    update epsilon and noise
        self.epsilon *= self.epsilon_decay
        self.noise.reset()
    


    def reset(self):
        self.noise.reset()
    
    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
            
    def save_models(self):
        """ Save models weights """
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()
        
    def load_models(self):
        """ Load models weights """
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()            