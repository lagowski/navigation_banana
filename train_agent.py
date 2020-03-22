from unityagents import UnityEnvironment
import numpy as np
import random
import torch
from collections import deque
import time
import matplotlib.pyplot as plt
from dqn_agent import Agent

EPISODES = 2000
MAX_T = 1000
EPS_START = 1.0

EPS_END = 0.01
EPS_DECAY = 0.995

STOP_REWARD = 13.0


def dqn(n_episodes=EPISODES, max_t=MAX_T, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        # state = env.reset()
        
        # RESET ENV
        env_info = env.reset(train_mode=True)[brain_name]
        
        # GET STATE
        state = env_info.vector_observations[0]
        
        score = 0
        for t in range(max_t):
            
            # DEFINE ACTION
            action = agent.act(state, eps)
            
            # MAKE ACTION !  
            env_info = env.step(action.astype(int))[brain_name]
            
            # GET NEXT /ACTUAL STATE
            next_state = env_info.vector_observations[0]
            
            # GET ACTUAL REWARD
            reward = env_info.rewards[0]
            
            # CHECK if LAST EPISODE
            done = env_info.local_done[0]
            
                   
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=STOP_REWARD:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores




if __name__ == '__main__':
    # Initialize env 
    env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
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
    scores = dqn()

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    fig.savefig('./img/Figure_1.png')
    env.close()