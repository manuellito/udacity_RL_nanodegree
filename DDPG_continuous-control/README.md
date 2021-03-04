[//]: # (Image References)

[image1]: media/reacher.png "Trained Agent"
[image2]: media/kernel.png  "Kernel"
[image3]: media/graph.png   "Reward Graph"

# Continuous Control implementation in the Unity environment

## Introduction

This project is an example of an implementation of the Continuous Control to teach to an agent (or several here) a mechanic arm to reach a target.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Requirement

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

3. Install Unity Agents

        pip install unityagents==0.4.0

4.  Install PyTorch
   
        pip install torch==0.4.0

5.  Clone this repositry

        git clone https://github.com/manuellito/DDPG_continuous-control.git

6. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  

        python -m ipykernel install --user --name drlnd --display-name "drlnd"

7. Start Jupyter

        jupyter notebook

8. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

    ![Kernel][image2]

## Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
   - **_Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 20 agents) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)



2. Place the file in the DRLND GitHub repository, in the `Data/` folder in this reposity, and unzip (or decompress) the file. 
3. In the  `Continuous_Control.ipynb` replace the following line by the path where you just install the application.
   
        env = UnityEnvironment(file_name='Data/Reacher_Linux/Reacher.x86_64')

    You don't need to modify it if you are using a Linux system.

## Instructions

Execute instructions in `Continuous_Control.ipynb` to get started with training your own agent!  
It takes about 200 episodes to train the agent to reach the goal of 30 points on average over the last 100 episodes.

The training graph, should be like that:

![Reward Graph][image3]

## Test you network

If you want to test the network you've just trained, run the "Let's play the agent" box at the end.
