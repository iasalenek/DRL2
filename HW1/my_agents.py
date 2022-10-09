from world.realm import Realm
from world.envs import OnePlayerEnv, VersusBotEnv, TwoPlayerEnv
from world.utils import RenderedEnvWrapper
from world.map_loaders.base import MixedMapLoader
from world.map_loaders.single_team import SingleTeamLabyrinthMapLoader, SingleTeamRocksMapLoader
from world.map_loaders.two_teams import TwoTeamLabyrinthMapLoader, TwoTeamRocksMapLoader
from world.scripted_agents import ScriptedAgent


import abc
import random
import numpy as np
import copy
from collections import deque

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from world.my_rewards import (calc_distance_map, 
                              calc_closeness, 
                              get_reward_1, 
                              compute_observation)
                              

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 10000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
BUFFR_SIZE = 10000
EVAL_EVERY = 100

DEVICE = 'cpu'


class DQN(ScriptedAgent):

    def __init__(self):
        
        self.steps = 0
        self.distance_map = None

        # Torch model
        torch.manual_seed(0)
        self.model = nn.Sequential(
            nn.Conv2d(5, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1600, 400),
            nn.ReLU(),
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.Linear(100, 25)).requires_grad_(True).to(DEVICE)

        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.target_model = copy.deepcopy(self.model).requires_grad_(False)
        self.seed = random.seed(0)
        self.buffer = deque(maxlen=BUFFR_SIZE)

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.buffer.append(transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        sample = random.sample(dqn.buffer, BATCH_SIZE)
        
        observations = []
        actions = []
        rewards = []
        dones = []
        next_observations = []

        for experience in sample:

            state, distance_map, action, next_state, done, env = experience

            # Считаем наблюдения из состояний
            obs = compute_observation(state)
            next_obs = compute_observation(next_state)

            # Считаем награду
            if not done:
                reward = get_reward_1(env, state, action, distance_map)
            else:
                reward = 0.0
                    
            # Костыль на всякий случай
            if np.isnan(reward):
                reward = 0.0
            
            observations.append(obs)
            actions.append(action)
            next_observations.append(next_obs)
            dones.append([done])
            rewards.append(reward)
        
        return [torch.Tensor(np.array(i)) for i in [observations, actions, next_observations, rewards, dones]]


    def train_step(self, batch):
        observations, actions, next_observations, rewards, dones = batch

        ind_0 = np.repeat(np.arange(16)[:, None], 5, axis = 1)
        ind_1 = np.repeat(np.arange(5)[None], 16, axis = 0)
        
        self.optimizer.zero_grad()

        Q = self.model(observations).view(BATCH_SIZE, 5, 5)[ind_0, ind_1, actions.to(int)]

        Q_next = torch.amax(self.target_model(next_observations).view(BATCH_SIZE, 5, 5), dim=2) * torch.logical_not(dones)
        
        loss = F.mse_loss(Q, rewards[:, None] + GAMMA * Q_next)
        loss.backward()

        self.optimizer.step()

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or 
        # assign a values of network parameters via PyTorch methods.
        self.target_model = copy.deepcopy(self.model).requires_grad_(False)
    
    def get_actions(self, state, team=0):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        obs = compute_observation(state)
        obs = np.expand_dims(obs, axis=0)
        action = self.model(torch.Tensor(obs)).view(5, 5).argmax(axis = 1)
        return action.to(int).cpu().detach().tolist()

    def update(self, transition):
            # You don't need to change this
            self.consume_transition(transition)
            if self.steps % STEPS_PER_UPDATE == 0:
                batch = self.sample_batch()
                self.train_step(batch)
            if self.steps % STEPS_PER_TARGET_UPDATE == 0:
                self.update_target_network()
            self.steps += 1

    def save(self):
        torch.save(self.model, "agent.pkl")


def evaluate_policy(agent, episodes=5):

    env = OnePlayerEnv(Realm(
        MixedMapLoader((SingleTeamLabyrinthMapLoader(), SingleTeamRocksMapLoader())),
        1
    ))

    returns = []
    eaten = []

    for _ in range(episodes):
        done = False
        total_reward = 0.
        total_eaten = 0
        state, info = env.reset()
        agent.reset(state, 0)
        distance_map = calc_distance_map(state)
        
        while not done:
            action = agent.get_actions(state, team=0)
            reward = get_reward_1(copy.deepcopy(env), state, action, distance_map)
            # Костыль на всякий случай
            if np.isnan(reward):
                reward = 0.0

            next_state, done, info = env.step(action)
            
            total_reward += reward
            total_eaten += len(info['eaten'])

        returns.append(total_reward)
        eaten.append(total_eaten)
        
    return returns, eaten


if __name__ == "__main__":
    env = OnePlayerEnv(Realm(
        MixedMapLoader((SingleTeamLabyrinthMapLoader(), SingleTeamRocksMapLoader())),
        1
    ))
    dqn = DQN()
    eps = 0.1
    state, info = env.reset()
    distance_map = calc_distance_map(state)

    for _ in range(INITIAL_STEPS):

        action = [np.random.randint(5) for i in range(5)]
        env_deepcopy = copy.deepcopy(env)
        next_state, done, info = env.step(action)
        dqn.consume_transition((state, distance_map, action, next_state, done, env_deepcopy))
        
        if not done:
            state = next_state 
        else:
            state, info = env.reset()
            distance_map = calc_distance_map(state)

    for i in range(TRANSITIONS):
        #Epsilon-greedy policy
        if random.random() < eps:
            action = [np.random.randint(5) for i in range(5)]
        else:
            action = dqn.get_actions(state)

        env_deepcopy = copy.deepcopy(env)
        next_state, done, info = env.step(action)
        dqn.update((state, distance_map, action, next_state, done, env_deepcopy))
        
        if not done:
            state = next_state 
        else:
            state, info = env.reset()
            distance_map = calc_distance_map(state)

        if (i + 1) % (EVAL_EVERY) == 0:
                rewards, eaten = evaluate_policy(dqn, 5)
                print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
                print(f"Step: {i+1}, Eaten mean: {np.mean(eaten)}, Eaten std: {np.std(eaten)}\n")
                dqn.save()
