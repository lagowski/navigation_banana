from unityagents import UnityEnvironment
import numpy as np
import random
import torch
from collections import deque
import time
import matplotlib.pyplot as plt
from dqn_agent import Agent

MAX_T = 1000
EPISODES = 3

if __name__ == '__main__':
    # Initialize env 
    env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name]
    print("Nº agents", len(env_info.agents))

    action_size = brain.vector_action_space_size
    print("Nº actions", action_size)

    # GET STATE FROM ENV -> vector_observations[0]
    state = env_info.vector_observations[0]
    state_size = len(state)

    print("States", state)
    print("---")
    print("State size", state_size)


    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

    for i in range(EPISODES):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        score = 0

        for i in range(MAX_T):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            time.sleep(0.1)
            if done:
                break
        print("Episode: ", i+1, " score ", score)
    env.close()    
