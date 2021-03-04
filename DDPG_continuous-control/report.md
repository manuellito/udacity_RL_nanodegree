[//]: # (Image References)
[image1]: media/ddpg_algo.png "algo"
[image2]: media/actor.png "Actor"
[image3]: media/critic.png "Critic"
[image4]: media/graph.png "Graph Mean"

# Continuous Control implementation in the Unity environment

## Scope

The objective of this repository is to propose a solution for the resolution of the reacher environment.

## Description of the problem

This project uses the [Unity](https://github.com/Unity-Technologies/ml-agents) environment to manage mechanics arms. The goal is to follow a target with these arms.

In order to successfully complete the training, the agent must have an average of 30 points over the last 100 episodes.

## Description of the environment

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Learning architecture

This solution implements a DDPG agent from [Deepmind paper](https://arxiv.org/pdf/1509.02971.pdf).

From this paper I implemented the aglorithm:

![Algo][image1]


In this implementation I've used a replay buffer with 2 neural networks: a Critic and Agent one.

Each neural network uses 2 hidden layers with number of units come from the previous papers: 400 and 300.

It is described as follows

_Actor Network_

![Network][image2]

_Critic Network_

![Network][image3]

The hyperparameters used are the following:

|  Hyperparameters |  value | description  |
|---|---|---|
| Buffer size  | 1000000  | Number of episods in the replay buffer
| Batch Size  | 64  |  Number of episods in each batch | 
|  Lr actor | 0.001  | Learning rate for Agent network |
|  Lr critic | 0.0001  | Learning rate for Critic network |
| Gamma  | 0.99  |  Discount value |
|  Tau | 0.001  | Factor to update the target network  |
|  Every update | 2  | How often the target network is updated  |


The implementation is built around the following files:

* utils.py
  
  Replay buffer implementation to save and retrieve previous played episodes. They stores as a tuples of (s, a, r, s', d).

  In this file, I've implemented Ornstein-Ulhenbeck Noise in order to add some noise in action decision.

* model.py

  Implements neurals network, layers and units for actor (ActorNetwork) and critic (CriticNetwork)

* ddpg_agent.py
  
  Implements the agent which has to move around in an environment where it does not know the rules and must learn them from his experience.

  The implementation manage 20 agents (or more for sure)

* Continuous_Control.ipynb
  
  Main file which manage the link between agents and its environment

## Results

With a fairly simple implementation, the network does not do so badly.

It takes less than 300 episodes to resolve the environment.

Here's an example of rewards:

![Graph Mean][image4]

## Ideas for future work

Here are the improvements I could make later on to improve performance or learning:

* A better fine tuning hyperparameters in order to improcve the efficiency.

* Implement other networks such as `D4PG` and `A3C` in order to compare performance.
