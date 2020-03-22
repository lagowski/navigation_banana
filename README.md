[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    
2. Place the file in the  `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

It's required use of python 3.6
```
conda create --name myenv python=3.6
```
And activate 

```
source activate myenv
```

Use of tensorflow 1.7.1 required by unityagents

```
pip install tensorflow==1.7.1
pip install unityagents
```

Use of pytoch
```
pip install pytorch
```

##### For train run
```
python train_agent.py
```

![Scores](img/Figure_1)

##### Play with agent

```
python play_agent.py
```

