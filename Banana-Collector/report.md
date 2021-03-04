[//]: # (Image References)
[image1]: media/Network.png "Network"
[image2]: media/graph_mean.png "Graph Mean"
[image3]: media/example_compare.png "Example Compare"

# banana collector - Deep Reinforcment Learning Udacity Nanodegree

## Scope

The objective of this repository is to propose a solution for the resolution of the banana collector

## Description of the problem

This project uses the [Unity](https://github.com/Unity-Technologies/ml-agents) environment to drive an agent in a closed environment.

The objective of the agent is to collect a maximum of yellow bananas.

Each yellow banana earns +1 reward to the agent, while each blue banana earns -1 of reward.

In order to successfully complete the training, the agent must have an average of 13 points over the last 100 episodes.

## Description of the environment

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

Each episode is done if the environment return the done flag or if a max step is reach (manage by the program)

## Learning architecture

This solution implements [Deepmind paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)

In order to stabilize the network, we find the solutions described by Deepmind, namely, the 2 neural networks and the replay buffer.

The network consists of 3 layers. The first contains 64 units, the second 64 and the last 4.

It is described as follows

![Network][image1]

The hyperparameters used are the following:

|  Hyperparameters |  value | description  |
|---|---|---|
| Buffer size  | 500000  | Number of episods in the replay buffer
| Batch Size  | 64  |  Number of episods in each batch | 
| Gamma  | 0.99  |  Discount value |
|  Tau | 0.001  | Factor to update the target network  |
|  lr | 0.0005  | Learning rate  |
|  Every update | 4  | How often the target network is updated  |
| Eps start  | 1.0  |  First value of epsilon |
|  Eps end |  0.1 |  Minimum value for epsilon |
|  eps decay | 0.995  | Epsilon decay  |

The implementation is built around the following files:

* replaybuffer.py
  
  Replay buffer implementation to save and retrieve previous played episodes. They stores as a tuples of (s, a, r, s', d)

* model.py

  Implements neurals network, layers and units

* dqn_agent.py
  
  Implements the agent which has to move around in an environment where it does not know the rules and must learn them from his experience.

* Navigation.ipynb
  
  Main file which manage the ling between the agent and its environment

## Results

With a fairly simple implementation, the network does not do so badly.

It takes between 600 and 700 episodes to resolve the environment and less than 10 minutes.

Each execution gives a different result, so they are orders of magnitude.

Here's an example of rewards:

![Graph Mean][image2]

## Benchmark

I used two tools in order to modify the hyperparameters and see their impact on the learning of the agent.

[Wandb](https://www.wandb.com/) is usefull for graphs and [Jovian](https://www.jovian.ai/) is usefull for compare hyperparameters.

You can find projects here:
 
 * [https://wandb.ai/batmanu/udacity_banana-dqn](https://wandb.ai/batmanu/udacity_banana-dqn)
 * [https://jovian.ai/manu-farcy/udacity-banana-dqn/compare](https://jovian.ai/manu-farcy/udacity-banana-dqn/compare)

I tried several values for hyperparameters or a network with more layers or with more units.

The `Deeply` test uses one more layer, 64 units.

The tests with more neurons, used 256 units for the layers using 64 units. 

This is an example of rewards with differents values for hyperparameters

![Example Compare][image3]

`Pypical Case` and `Final Run` use the same hyperparameters (the values used in the code) to show the impact of two executions with the same parameters.

Here is the table describing for each hyperparameter value, the impact on the learning time and the number of episodes needed to resolve the environment .

Remember that 2 executions with the same hyperparameters will not give exactly the same results. If the difference is small, it can therefore be concluded that the impact is limited.


|  Run name |  Batch size | Buffer size  | Gamma | Learning rate | Tau | Every update | Total episodes | Running time
|---|---|---|---|---|---|---|---|---|
|Typical Case |	64	| 500000 | 0.99 | 0.0005 | 0.001 | 4 | 530| 0:06:40.247	|
|elephant memory | 64 | 1000000 | 0.99 | 0.0005 | 0.001 | 4 | 596| 0:07:48.640 |
|Large batch size | 256 | 500000 | 0.99 | 0.0005 | 0.001 | 4 | 676 | 0:11:46.687 |
|Low batch size | 24 | 500000 | 0.99 | 0.0005 | 0.001 | 4 | 622 | 0:07:25.661 |
|Higher tau | 64 | 500000 | 0.99 | 0.0005 | 0.1 | 4 | 1369 | 0:17:40.904 |
|Lower lr | 64 | 500000 | 0.99 | 0.00005 | 0.001 | 4 | 689 | 0:08:42.641 |
|Higher update_every | 64 | 500000 | 0.99 | 0.0005 | 0.001 | 12 | 890 | 0:09:45.020 |
|Lower update_every | 64 | 500000 | 0.99 | 0.0005 | 0.001 | 2 | 632 | 0:10:08.595 |
|More neurons | 64 | 500000 | 0.99 | 0.0005 | 0.001 | 4 | 830 | 0:14:18.044 |
|Deeply | 64 | 500000 | 0.99 | 0.0005 | 0.001 | 4 | 514 | 0:06:54.234 |
|Deeply with more units | 64 | 500000 | 0.99 | 0.0005 | 0.001 | 4 | 576 | 0:12:15.461 |
|Final run | 64 | 500000 | 0.99 | 0.0005 | 0.001 | 4 | 640 | 0:08:57.490 |

## Ideas for future work

Here are the improvements I could make later on to improve performance or learning:

* In order to compare performance, I would like to test, for the replay buffer, the use of an indexed array, rather than using the deque object.

* Implement the `prioritized experience replay'. The impact on the code should be minor, but the result could be significant.

* Implement other networks such as `Double DQN` and `Dueling DQN`.

* Rainbow implementation, which would take previous implementations that have had a significant impact on performance or learning and combine them into a single solution.

