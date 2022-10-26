from world.realm import Realm
from world.envs import OnePlayerEnv, VersusBotEnv, TwoPlayerEnv
from world.utils import RenderedEnvWrapper
from world.map_loaders.base import MixedMapLoader
from world.map_loaders.single_team import SingleTeamLabyrinthMapLoader, SingleTeamRocksMapLoader
from world.map_loaders.two_teams import TwoTeamLabyrinthMapLoader, TwoTeamRocksMapLoader
from world.scripted_agents import ScriptedAgent

from world.scripted_agents import ClosestTargetAgent, BrokenClosestTargetAgent

import abc
import argparse
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
from my_utils.dqn_utils import compute_observation, vs_agent_reward
from nets import Slava_net, Simple_conv

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='simple')
parser.add_argument('--batch_norm', action='store_true')
parser.add_argument('--no-batch_norm', dest='batch_norm', action='store_false')
parser.set_defaults(batch_norm=False)
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--transitions', type=int, default=100000)
parser.add_argument('--buffer_size', type=int, default=10000)
parser.add_argument('--initial_steps', type=int, default=1000)
parser.add_argument('--step_per_update', type=int, default=4)
parser.add_argument('--target_update', type=int, default=4000)
parser.add_argument('--num_predators', type=int, default=5)
parser.add_argument('--epsilon', type=float, default=0.2)
parser.add_argument('--eval_every', type=int, default=10000)
parser.add_argument('--episodes', type=int, default=10)

args = parser.parse_args()                       

NET = args.net
BATCH_NORM = args.batch_norm
DROPOUT = args.dropout

GAMMA = 0.99
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
DEVICE = args.device
TRANSITIONS = args.transitions
BUFFR_SIZE = args.buffer_size
INITIAL_STEPS = args.initial_steps
STEPS_PER_UPDATE = args.step_per_update
STEPS_PER_TARGET_UPDATE = args.target_update
NUM_PREDATORS = args.num_predators
EPSILON = args.epsilon
EVAL_EVERY = args.eval_every
EPISODES = args.episodes

print(f'NET: {NET}')
print(f'BATCH_NORM: {BATCH_NORM}')
print(f'DROPOUT: {DROPOUT}')

print(f'BATCH_SIZE: {BATCH_SIZE}')
print(f'LEARNING_RATE: {LEARNING_RATE}')
print(f'DEVICE: {DEVICE}')
print(f'TRANSITIONS: {TRANSITIONS}')
print(f'BUFFR_SIZE: {BUFFR_SIZE}')
print(f'INITIAL_STEPS: {INITIAL_STEPS}')
print(f'STEPS_PER_UPDATE: {STEPS_PER_UPDATE}')
print(f'STEPS_PER_TARGET_UPDATE: {STEPS_PER_TARGET_UPDATE}')
print(f'NUM_PREDATORS: {NUM_PREDATORS}')
print(f'EPSILON: {EPSILON}')
print(f'EVAL_EVERY: {EVAL_EVERY}')
print(f'EPISODES: {EPISODES}')

reward_func = vs_agent_reward

if NET == 'simple':
    model = Simple_conv(5, batch_norm = BATCH_NORM, dropout=DROPOUT)
elif NET == 'slava':
    model = Slava_net(5, batch_norm = BATCH_NORM, dropout=DROPOUT)

class DQN(ScriptedAgent):

    def __init__(self, id: int = 0):
        
        self.steps = 0

        self.model = model.requires_grad_(True).to(DEVICE)

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

            state, action, next_state, done, info, distance_map = experience

            # Оставляем только действие одного агента

            agent_id = np.random.randint(5)
            action = action[agent_id]

            # Считаем наблюдения из состояний
            obs = compute_observation(agent_id, state, distance_map)
            next_obs = compute_observation(agent_id, next_state, distance_map)

            ###
            # import time
            # import matplotlib.pyplot as plt
            # plt.imshow(obs[3], cmap='Greys')
            # plt.show()
            # time.sleep(5)
            # plt.imshow(obs[4], cmap='Greys')
            # plt.show()
            # time.sleep(5)
            # plt.imshow(next_obs[3])
            # plt.show()
            # time.sleep(5)
            ###

            # Считаем награду
            reward = reward_func(state, agent_id, next_state, info, distance_map)

            observations.append(obs)
            actions.append(action)
            next_observations.append(next_obs)
            dones.append([done])
            rewards.append([reward])

        return [torch.Tensor(np.array(i)).to(DEVICE) for i in [
                    observations, actions, next_observations, rewards, dones]]


    def train_step(self, batch):
        observations, actions, next_observations, rewards, dones = batch

        # ####
        # print(f'obs shape: {observations.shape}')
        # print(f'actions shape: {actions.shape}')
        # print(f'next_obs shape: {next_observations.shape}')
        # print(f'dones shape: {dones.shape}')
        # print(f'rewards shape: {rewards.shape}')
        # time.sleep(10)
        # ###
    
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
    
    def get_actions(self, observations, team=0):
        
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


def evaluate_policy(agent, enemy, episodes=5):

    agent.model.eval()

    env = TwoPlayerEnv(Realm(
        MixedMapLoader((TwoTeamLabyrinthMapLoader(), TwoTeamRocksMapLoader())),
        2))

    scores_0 = []
    scores_1 = []

    for _ in range(episodes):
        done = False
        data = env.reset()
        state_0, info_0 = data[0]
        state_1, info_1 = data[1]

        agent.reset(state_0, 0)
        enemy.reset(state_1, 0)

        distance_map = calc_distance_map(state_0)
        
        while not done:

            # Собираем наблюдения всех агентов
            observations = np.array([compute_observation(
                agent_id, state_0, distance_map) for agent_id in range(5)])

            actions_0 = agent.get_actions(observations, team=0)
            actions_1 = enemy.get_actions(state_1, team=0)

            next_data = env.step(actions_0, actions_1)
            next_state_0, done, info_0 = next_data[0]
            next_state_1, done, info_1 = next_data[1]

            state_0 = next_state_0
            state_1 = next_state_1

        scores_0.append(info_0['scores'][0])
        scores_1.append(info_0['scores'][1])

    agent.model.train()
        
    return scores_0, scores_1


def main():
    env = TwoPlayerEnv(Realm(
        MixedMapLoader((TwoTeamLabyrinthMapLoader(), TwoTeamRocksMapLoader())),
        2))
    agent = DQN()
    enemy = ClosestTargetAgent()

    data = env.reset()
    state_0, info_0 = data[0]
    state_1, info_1 = data[1]

    agent.reset(state_0, 0)
    enemy.reset(state_1, 0)

    distance_map = calc_distance_map(state_0)

    for _ in range(INITIAL_STEPS):

        actions_0 = [np.random.randint(5)] * 5
        actions_1 = enemy.get_actions(state_1, team=0)
        
        next_data = env.step(actions_0, actions_1)
        next_state_0, done, info_0 = next_data[0]
        next_state_1, done, info_1 = next_data[1]

        agent.consume_transition((state_0, actions_0, next_state_0, done, info_0, distance_map))
        
        if not done:
            state_0 = next_state_0
            state_1 = next_state_1
        else:
            data = env.reset()
            state_0, info_0 = data[0]
            state_1, info_1 = data[1]

            agent.reset(state_0, 0)
            enemy.reset(state_1, 0)

            distance_map = calc_distance_map(state_0)


    for i in tqdm(range(TRANSITIONS)):
        # Собираем наблюдения всех агентов
        observations = np.array([compute_observation(
            agent_id, state_0, distance_map) for agent_id in range(5)])

        # epsilon greedy
        if random.random() < EPSILON:
            actions_0 = [np.random.randint(5)] * 5
        else:
            actions_0 = agent.get_actions(observations)

        actions_1 = enemy.get_actions(state_1, team=0)

        next_data = env.step(actions_0, actions_1)
        next_state_0, done, info_0 = next_data[0]
        next_state_1, done, info_1 = next_data[1]

        agent.update((state_0, actions_0, next_state_0, done, info_0, distance_map))
        
        if not done:
            state_0 = next_state_0
            state_1 = next_state_1
        else:
            data = env.reset()
            state_0, info_0 = data[0]
            state_1, info_1 = data[1]

            agent.reset(state_0, 0)
            enemy.reset(state_1, 0)

            distance_map = calc_distance_map(state_0)

        if (i + 1) % (EVAL_EVERY) == 0:
            scores_0, scores_1 = evaluate_policy(agent, enemy, EPISODES)
            print(f"\nscore_0 mean: {np.mean(scores_0)}, score_0 var: {np.var(scores_0)}\nscore_1 mean: {np.mean(scores_1)}, score_1 var: {np.var(scores_1)}\nwinrate: {np.mean(scores_0 > scores_1)}")
            agent.save()


if __name__ == "__main__":
    main()