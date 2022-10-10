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
from tqdm import tqdm
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from my_utils import (calc_distance_map, 
                      calc_closeness, 
                      get_reward_1, 
                      compute_observation)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--initial_steps', type=int, default=1024)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--eval_every', type=int, default=1000)
parser.add_argument('--transitions', type=int, default=10000)
parser.add_argument('--target_update', type=int, default=200)
parser.add_argument('--episodes', type=int, default=50)
parser.add_argument('--step_per_update', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_predators', type=int, default=5)
parser.add_argument('--epsilon', type=float, default=0.2)

args = parser.parse_args()                       

GAMMA = 0.99
BUFFR_SIZE = 10000
INITIAL_STEPS = args.initial_steps
TRANSITIONS = args.transitions
STEPS_PER_UPDATE = args.step_per_update
STEPS_PER_TARGET_UPDATE = args.target_update
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
EVAL_EVERY = args.eval_every
EPISODES = args.episodes
DEVICE = args.device
NUM_PREDATORS = args.num_predators
EPSILON = args.epsilon

print(f'BATCH_SIZE: {BATCH_SIZE}')
print(f'INITIAL_STEPS: {INITIAL_STEPS}')
print(f'LEARNING_RATE: {LEARNING_RATE}')
print(f'DEVICE: {DEVICE}')
print(f'EPSILON: {EPSILON}')
print(f'STEPS_PER_UPDATE: {STEPS_PER_UPDATE}')
print(f'STEPS_PER_TARGET_UPDATE: {STEPS_PER_TARGET_UPDATE}')
print(f'NUM_PREDATORS: {NUM_PREDATORS}')


class DQN(ScriptedAgent):

    def __init__(self, 
        num_predators: int = 5):
        
        self.steps = 0
        self.num_predators = NUM_PREDATORS

        # Torch model
        torch.manual_seed(0)
        # self.model = nn.Sequential(
        #     nn.Conv2d(self.num_predators, 64, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.AvgPool2d(2),
        #     nn.Conv2d(64, 64, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.AvgPool2d(2),
        #     nn.Conv2d(64, 64, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.AvgPool2d(2),
        #     nn.Flatten(),
        #     nn.Linear(1600, 400),
        #     nn.ReLU(),
        #     nn.Linear(400, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 5 * self.num_predators)).requires_grad_(True).to(DEVICE)
        model = nn.Sequential(
            nn.Conv2d(5, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1600, 400),
            nn.ReLU(),
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.Linear(100, 5 * 5)).requires_grad_(True).to(DEVICE)

        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.target_model = copy.deepcopy(self.model).requires_grad_(False).to(DEVICE)
        self.buffer = deque(maxlen=BUFFR_SIZE)

    def consume_transition(self, transition):
        self.buffer.append(transition)

    def sample_batch(self):
        sample = random.sample(dqn.buffer, BATCH_SIZE)
        
        observations = []
        actions = []
        rewards = []
        dones = []
        next_observations = []

        for experience in sample:

            state, distance_map, action, next_state, done, env = experience

            # Считаем наблюдения из состояний
            obs = compute_observation(state, self.num_predators)

            #print(obs[:, 20, 20])

            next_obs = compute_observation(next_state, self.num_predators)

            # Считаем награду
            if not done:
                reward = get_reward_1(env, state, action, distance_map, self.num_predators)
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

        # Нормировка наград
        # rewards = (rewards - np.mean(reward)) / np.sqrt(np.var(rewards))
        
        return [torch.Tensor(np.array(i)).to(DEVICE) for i in [observations, actions, next_observations, rewards, dones]]


    def train_step(self, batch):
        observations, actions, next_observations, rewards, dones = batch

        ind_0 = np.repeat(np.arange(BATCH_SIZE)[:, None], self.num_predators, axis = 1)
        ind_1 = np.repeat(np.arange(self.num_predators)[None], BATCH_SIZE, axis = 0)
        
        self.optimizer.zero_grad()

        Q = self.model(observations).view(BATCH_SIZE, self.num_predators, 5)[ind_0, ind_1, actions.to(int)]

        Q_next = torch.amax(self.target_model(next_observations).view(BATCH_SIZE, self.num_predators, 5), dim=2) * torch.logical_not(dones)

        # print(Q_next)
        
        loss = F.mse_loss(Q, rewards[:, None] + GAMMA * Q_next)

        print(loss)

        loss.backward()

        self.optimizer.step()

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or 
        # assign a values of network parameters via PyTorch methods.
        self.target_model = copy.deepcopy(self.model).requires_grad_(False).to(DEVICE)
    
    def get_actions(self, state, team=0):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        obs = compute_observation(state, self.num_predators)
        obs = np.expand_dims(obs, axis=0)
        action = self.model(torch.Tensor(obs).to(DEVICE)).view(self.num_predators, 5).argmax(axis = 1)

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
        MixedMapLoader((SingleTeamLabyrinthMapLoader(), 
        # SingleTeamRocksMapLoader()
        )),
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
            reward = get_reward_1(copy.deepcopy(env), state, action, distance_map, NUM_PREDATORS)
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
        MixedMapLoader((SingleTeamLabyrinthMapLoader(),
        # SingleTeamRocksMapLoader()
        )),
        1
    ))
    dqn = DQN(num_predators=NUM_PREDATORS)
    state, info = env.reset()
    distance_map = calc_distance_map(state)

    for _ in range(INITIAL_STEPS):

        action = [np.random.randint(5) for i in range(NUM_PREDATORS)]
        env_deepcopy = copy.deepcopy(env)
        next_state, done, info = env.step(action)
        dqn.consume_transition((state, distance_map, action, next_state, done, env_deepcopy))
        
        if not done:
            state = next_state 
        else:
            state, info = env.reset()
            distance_map = calc_distance_map(state)

    for i in tqdm(range(TRANSITIONS)):
        #Epsilon-greedy policy
        if random.random() < EPSILON:
            action = [np.random.randint(5) for i in range(NUM_PREDATORS)]
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
                rewards, eaten = evaluate_policy(dqn, EPISODES)
                print(f"\n Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
                print(f"Step: {i+1}, Eaten mean: {np.mean(eaten)}, Eaten std: {np.std(eaten)}\n")
                dqn.save()
