[//]: # (Image References)

[image1]: media/banana_example.gif "Trained Agent"
[image2]: media/kernel.png  "Kernel"
[image3]: media/graph.png   "Reward Graph"

# Deep Q-Network implementation in the Unity environment

### Introduction

This project is an example of an implementation of the Deep Q-Network to teach an agent to move in a finite environment following some rules he doesn't know.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- move forward.
- move backward.
- turn left.
- turn right.

The task is episodic. The environment is resolved when the agent scores 13 on the average of the last 100 episodes.


### Requirement

In order to train your own agent, you need these feature:

 - Python 3.6
 - PyTorch 0.4
 - UnityAgent

In order to install them follow these steps:

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/) [(download)](https://docs.anaconda.com/download/)

2. Create an activate a new environment:
   
    * Linux or Mac:

            conda create --name drlnd python=3.6

            source activate drlnd

    * Windows:

            conda create --name drlnd python=3.6 
        
            activate drlnd
  
3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).

4. Install Unity Agents

        pip install unityagents==0.4.0

5.  Install PyTorch
   
        pip install torch==0.4.0

6.  Clone this repositry

        git clone https://github.com/manuellito/udacity_RL_nanodegree.git

7. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  

        python -m ipykernel install --user --name drlnd --display-name "drlnd"

8. Start Jupyter

        jupyter notebook

9. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

    ![Kernel][image2]

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `Data/` folder in this reposity, and unzip (or decompress) the file. 
3. In the  `Navigation.ipynb` replace the following line by the path where you just install the application.
   
        env = UnityEnvironment(file_name="./Data/Banana_Linux/Banana.x86_64")

    You don't need to modify it if you are using a Linux system.

### Instructions

Execute instructions in `Navigation.ipynb` to get started with training your own agent!  
It takes about 500 episodes to train the agent to reach the goal of 13 points on average over the last 100 episodes.

The training graph, should be like that:

![Reward Graph][image3]

### Test your network

If you want to test the network you've just trained, run the "Test your network" box at the end.

### Side note

In order to follow the evolution of the benches of each version, this implementation uses [Wandb](https://wandb.ai/) and [Jovian](https://jovian.ai/).

If you want to use it without that, remove all references in `Navigation.ipynb` and `dqn_agent.ipynb`.

Results can be fiund here:

* https://wandb.ai/batmanu/udacity_banana-dqn
* https://jovian.ai/manu-farcy/udacity-banana-dqn
  
## Reférences
* https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
* https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893
