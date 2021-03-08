from ddpg_agent import Agent
from utils import ReplayBuffer

import wandb
import jovian

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
LEARN_EVERY = 2
LEARN_NUMBER = 3
LR_ACTOR = 1e-3        # learning rate of the actor 
LR_CRITIC = 1e-2        # learning rate of the critic
GAMMA = 0.99          # discount factor
TAU = 1e-3              # for soft update of target parameters
EPSILON = 1.0
EPSILON_DECAY = 0.99
WEIGHT_DECAY = 0 #1e-2        # L2 weight decay
CLIPGRAD = .1

project_name = "rl_cooperative_tennis"
name = "Xmas"

class MADDPG:
    def __init__(self, state_size, action_size, seed = 42):
        super(MADDPG, self).__init__()

        self.agents = [Agent(state_size, action_size, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, agent_number=0, epsilon=EPSILON,
                             epsilon_decay=EPSILON_DECAY, weight_decay=WEIGHT_DECAY, clipgrad=CLIPGRAD), 
                       Agent(state_size, action_size, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, agent_number=1, epsilon=EPSILON,
                             epsilon_decay=EPSILON_DECAY, weight_decay=WEIGHT_DECAY, clipgrad=CLIPGRAD)]
        
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        # Init tracking of params
        wandb.login()
        wandb.init(project=project_name, name=name, config={"buffer_size": BUFFER_SIZE,
                                                          "batch_size": BATCH_SIZE,
                                                          "learn_every": LEARN_EVERY,
                                                          "learn_number": LEARN_NUMBER,
                                                          "lr_actor": LR_ACTOR,
                                                          "lr_critic": LR_CRITIC,
                                                          "gamma": GAMMA,
                                                          "tau": TAU,
                                                          "epsilon": EPSILON,
                                                          "epsilon_decay": EPSILON_DECAY,
                                                          "weight_decay": WEIGHT_DECAY,
                                                          "clipgrad": CLIPGRAD})
        
        jovian.log_hyperparams(project=project_name, name=name, config={"buffer_size": BUFFER_SIZE,
                                                          "batch_size": BATCH_SIZE,
                                                          "learn_every": LEARN_EVERY,
                                                          "learn_number": LEARN_NUMBER,
                                                          "lr_actor": LR_ACTOR,
                                                          "lr_critic": LR_CRITIC,
                                                          "gamma": GAMMA,
                                                          "tau": TAU,
                                                          "epsilon": EPSILON,
                                                          "epsilon_decay": EPSILON_DECAY,
                                                          "weight_decay": WEIGHT_DECAY,
                                                          "clipgrad": CLIPGRAD})

    def act(self, observations):
        """get actions from all agents in the MADDPG object"""

        actions = [agent.act(obs) for agent, obs in zip(self.agents,observations)]
        return actions

    def step(self, states, actions, rewards, next_states, dones, timestamp):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

            
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and timestamp % LEARN_EVERY == 0:
            for agent in self.agents:
                for _ in range(LEARN_NUMBER):
                    experiences = self.memory.sample()
                    agent.learn(experiences)
                
    def save(self):
        for agent in self.agents:
            agent.save_models()
            
    def get_project_name(self):
        return project_name
    
    def get_model_name(self):
        return name