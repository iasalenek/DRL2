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

from my_utils.common_utils import calc_distance_map
from my_utils.single_dqn_utils import (compute_observation_single,
                                       closest_n_reward)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--net', type=str, default='conv')
parser.add_argument('--transitions', type=int, default=50000)
parser.add_argument('--buffer_size', type=int, default=10000)
parser.add_argument('--initial_steps', type=int, default=10000)
parser.add_argument('--step_per_update', type=int, default=4)
parser.add_argument('--target_update', type=int, default=500)
parser.add_argument('--num_predators', type=int, default=5)
parser.add_argument('--epsilon', type=float, default=0.3)
parser.add_argument('--eval_every', type=int, default=1000)
parser.add_argument('--episodes', type=int, default=5)

args = parser.parse_args()                       

GAMMA = 0.99
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
DEVICE = args.device
NET = args.net
TRANSITIONS = args.transitions
BUFFR_SIZE = args.buffer_size
INITIAL_STEPS = args.initial_steps
STEPS_PER_UPDATE = args.step_per_update
STEPS_PER_TARGET_UPDATE = args.target_update
NUM_PREDATORS = args.num_predators
EPSILON = args.epsilon
EVAL_EVERY = args.eval_every
EPISODES = args.episodes

print(f'BATCH_SIZE: {BATCH_SIZE}')
print(f'LEARNING_RATE: {LEARNING_RATE}')
print(f'DEVICE: {DEVICE}')
print(f'NET: {NET}')
print(f'TRANSITIONS: {TRANSITIONS}')
print(f'BUFFR_SIZE: {BUFFR_SIZE}')
print(f'INITIAL_STEPS: {INITIAL_STEPS}')
print(f'STEPS_PER_UPDATE: {STEPS_PER_UPDATE}')
print(f'STEPS_PER_TARGET_UPDATE: {STEPS_PER_TARGET_UPDATE}')
print(f'NUM_PREDATORS: {NUM_PREDATORS}')
print(f'EPSILON: {EPSILON}')
print(f'EVAL_EVERY: {EVAL_EVERY}')
print(f'EPISODES: {EPISODES}')

reward_func = closest_n_reward

class singe_DQN(ScriptedAgent):

    def __init__(self):
        
        self.steps = 0

        if NET == 'conv':

            self.model = nn.Sequential(
                nn.Conv2d(4, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 2, 1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(800, 200),
                nn.ReLU(),
                nn.Linear(200, 50),
                nn.ReLU(),
                nn.Linear(50, 5)).requires_grad_(True).to(DEVICE)

        if NET == 'linear':

            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(4800, 1200),
                nn.ReLU(),
                nn.Linear(1200, 200),
                nn.ReLU(),
                nn.Linear(200, 50),
                nn.ReLU(),
                nn.Linear(50, 5),
            ).requires_grad_(True).to(DEVICE)

        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.target_model = copy.deepcopy(self.model).requires_grad_(False).to(DEVICE)
        self.buffer = deque(maxlen=BUFFR_SIZE)

    def consume_transition(self, transition):
        self.buffer.append(transition)

    def sample_batch(self):
        sample = random.sample(self.buffer, BATCH_SIZE)
        
        observations = []
        actions = []
        rewards = []
        dones = []
        next_observations = []

        for experience in sample:

            state, action, next_state, done, distance_map, info = experience

            # Оставляем только действие одного агента
            agent_id = np.random.randint(5)
            action_id = action[agent_id]

            # Считаем наблюдения из состояний
            obs = compute_observation_single(state, agent_id, distance_map)
            next_obs = compute_observation_single(next_state, agent_id, distance_map)

            # Считаем награду
            reward = reward_func(state, action, next_state, info, distance_map)
            
            observations.append(obs)
            actions.append(action_id)
            next_observations.append(next_obs)
            dones.append([done])
            rewards.append([reward])
        
        return [torch.Tensor(np.array(i)).to(DEVICE) for i in [
                    observations, actions, next_observations, rewards, dones]]


    def train_step(self, batch):
        observations, actions, next_observations, rewards, dones = batch

        self.optimizer.zero_grad()

        Q = self.model(observations)[torch.arange(BATCH_SIZE), actions.to(int)][:, None]

        Q_next = torch.amax(self.target_model(next_observations), dim=1, keepdim=True) * torch.logical_not(dones)

        loss = F.mse_loss(Q, rewards + GAMMA * Q_next)
        loss.backward()

        self.optimizer.step()

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or 
        # assign a values of network parameters via PyTorch methods.
        self.target_model = copy.deepcopy(self.model).requires_grad_(False).to(DEVICE)
    
    def get_actions(self, state, distance_map, team=0):
        
        observations = []
        for i in range(5):
            obs = compute_observation_single(state, i, distance_map)
            observations.append(obs)

        observations = np.array(observations)
        observations.shape
        
        actions = self.model(torch.Tensor(observations).to(DEVICE)).argmax(axis=1)
        return actions.to(int).cpu().detach().tolist()

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
            action = agent.get_actions(state, distance_map, team=0)
            next_state, done, info = env.step(action)
            reward = reward_func(state, action, next_state, info, distance_map)
            
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
    dqn = singe_DQN()
    state, info = env.reset()
    distance_map = calc_distance_map(state)

    for _ in range(INITIAL_STEPS):

        action = [np.random.randint(5) for i in range(5)]
        next_state, done, info = env.step(action)
        dqn.consume_transition((state, action, next_state, done, distance_map,info))
        
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
            action = dqn.get_actions(state, distance_map)

        next_state, done, info = env.step(action)
        dqn.update((state, action, next_state, done, distance_map,info))
        
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
