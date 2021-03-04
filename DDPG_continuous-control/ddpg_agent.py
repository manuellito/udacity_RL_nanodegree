import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from network import ActorNetwork, CriticNetwork
from utils import ReplayBuffer, OUActionNoise

import random

# Hyperparameters
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters     
UPDATE_EVERY = 2        # Learn every timestep

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, random_seed=42, num_agents=1):
        """Initialize Agent object.
        
        Params
        ====
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            lr_actor (float): Learning rate for actor model
            lr_critic (float): Learning Rate for critic model
            random_seed (int): Random seed
            num_agents (int): Number of agents
            
        return 
        ====
            None
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents

        # Initialize time step (for updating every hyperparameters["update_every"] steps)
        self.t_step = 0
        
        # Actor network
        self.actor = ActorNetwork(lr_actor, state_size, action_size, random_seed, name="actor")
        self.actor_target = ActorNetwork(lr_actor, state_size, action_size, random_seed, name="actor_target")
        
        self.soft_update(self.actor, self.actor_target, tau=1)

        # Critic network
        self.critic = CriticNetwork(lr_critic, state_size, action_size, random_seed, name="critic")
        self.critic_target = CriticNetwork(lr_critic, state_size, action_size, random_seed, name="critic_target")

        self.soft_update(self.critic, self.critic_target, tau=1)
        
        # Noise process
        self.noise = OUActionNoise(mu=np.zeros(action_size))

        # Replay buffer memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        # Support for multi agents learners
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        # Update timestep to learn
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = T.from_numpy(state).float().to(device)
        self.actor.eval()
        with T.no_grad():
            actions = self.actor(states).cpu().data.numpy()
        self.actor.train()

        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
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
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0) 
        self.critic.optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        # Minimize the loss
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic, self.critic_target, TAU)
        self.soft_update(self.actor, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def save_models(self):
        """ Save models weights """
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic_target.save_checkpoint()
        
    def load_models(self):
        """ Load models weights """
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic_target.load_checkpoint()
