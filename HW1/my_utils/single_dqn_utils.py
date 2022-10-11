import copy
import numpy as np
from my_utils.common_utils import calc_distance_map


def compute_observation_single(state: np.ndarray,
                               id: int,
                               distance_map: np.ndarray,
                               team: int = 0):

    y, x = np.where((state[:, :, 0] == team) * (state[:, :, 1] == id))
    state_centred = np.roll(np.roll(state, 20 - y, axis=0), 20 - x, axis=1)

    obs = np.zeros((40, 40, 4), dtype=int)

    #Наблюдения из state

    obs[:, :, 0][state_centred[:, :, 1] == -1] = 1 # Препятствия
    obs[:, :, 1][state_centred[:, :, 0] == 1] = 1  # Жертвы
    obs[:, :, 2][state_centred[:, :, 0] == 0] = 1  # Хищники

    # Distance map для агента

    for y1 in range(40):
        for x1 in range(40):
            obs[y1, x1, 3] = distance_map[y * 40 + x, y1 * 40 + x1]

    obs = np.transpose(obs, (2, 0, 1))
    
    return obs


def calc_closeness(state: np.ndarray, 
                   distance_map: np.ndarray):

    preys_team = np.max(state[:, :, 0])
    num_preys = np.sum(state[:, :, 0] == preys_team)
    preys_ind = np.where(state[:, :, 0] == preys_team)

    preys_dist = np.ones(num_preys) * 1601

    for i in range(5):

        y, x = np.where((state[:, :, 0] == 0) * (state[:, :, 1] == i))
        y, x = int(x), int(y)

        for j in range(num_preys):
            y1, x1 = preys_ind[0][j], preys_ind[1][j]

            dist = distance_map[x * 40 + y, x1 * 40 + y1]

            if dist < preys_dist[j]:
                preys_dist[j] = dist

    return -np.mean(preys_dist)


def get_reward_single(env,
                      state: np.ndarray, 
                      action: np.ndarray, 
                      distance_map: np.ndarray,
                      k: int = 1):
    
    cur_value = calc_closeness(state, distance_map)
    next_values = []

    for i in range(k):
        next_state, done, info = copy.deepcopy(env).step(action)
        if not done:
            next_values.append(
                # 10 * len(info['eaten']) + calc_closeness(next_state, distance_map))
                10 * len(info['eaten']))
        else:
            next_values.append(10 * len(info['eaten']))

    E_next_values = np.mean(next_values)
    #reward = (E_next_values - cur_value)
    if E_next_values != 0:
        # reward = cur_value / E_next_values - 1
        reward = E_next_values
    else:
        reward = 0


    return reward



def calc_closeness_id(state: np.ndarray,
                      distance_map: np.ndarray,
                      ids: np.ndarray):

    # Здесь команда охотников равна 0

    preys_team = np.max(state[:, :, 0])
    preys_ind = np.where(state[:, :, 0] == preys_team)
    predators_ind = np.where(state[:, :, 0] == 0)

    preys_dist = np.ones_like(ids) * 1601

    for y1, x1 in zip(*preys_ind):

        id = state[y1, x1][1]
        if id in ids:

            for y, x in zip(*predators_ind):
                dist = distance_map[y * 40 + x, y1 * 40 + x1]

                if dist < preys_dist[np.where(ids == id)]:
                    preys_dist[np.where(ids == id)] = dist

    return preys_dist



def closest_n_reward(state: np.ndarray,
                     action: np.ndarray, 
                     next_state: np.array,
                     info,
                     distance_map: np.ndarray,
                     n: int = 5):

    # Все 100 жертв
    all_ids = np.arange(100)
    all_dist = calc_closeness_id(state, distance_map, all_ids)

    # Только оставшиеся и достижимые жертвы
    ids = all_ids[all_dist < 1600]
    dist = all_dist[all_dist < 1600]

    # n ближайших жертв
    #ids_n = ids[np.argpartition(dist, min(n, len(dist) - 1))[:n]]
    ids_n = ids[np.argsort(dist)[:n]]
    dist_n = calc_closeness_id(state, distance_map, ids_n)
    cur_value = -np.mean(dist_n)

    # Считаем матожидание следущих состояний
    next_dist_n = calc_closeness_id(next_state, distance_map, ids_n)
    next_dist_n = next_dist_n[next_dist_n < 1600]
    next_value = -np.mean(next_dist_n)

    reward = next_value - cur_value + 10 * len(info['eaten'])

    ####
    # print(reward)
    ####

    return reward

#### Test

# from world.realm import Realm
# from world.envs import OnePlayerEnv, VersusBotEnv, TwoPlayerEnv
# from world.utils import RenderedEnvWrapper
# from world.map_loaders.base import MixedMapLoader
# from world.map_loaders.single_team import SingleTeamLabyrinthMapLoader, SingleTeamRocksMapLoader
# from world.map_loaders.two_teams import TwoTeamLabyrinthMapLoader, TwoTeamRocksMapLoader
# from world.scripted_agents import ScriptedAgent

# import matplotlib.pyplot as plt

# env = OnePlayerEnv(Realm(
#         MixedMapLoader((SingleTeamLabyrinthMapLoader(),
#         # SingleTeamRocksMapLoader()
#         )),
#         1
#     ))

## Тест награды

# action = [np.random.randint(5) for i in range(5)]
# closest_n_reward(env, state, action, distance_map)

## Визуализация однго id

# state_check = state.copy()
# state_check[(state_check[:, :, 0] == 1) * (state_check[:, :, 1] != 36)] = (-1, 0)
# state_check[(state_check[:, :, 1] == -1)] = [-2, -1]
# plt.imshow(state_check[:, :, 0])
# plt.show()

## График distance map для агента

# state, info = env.reset()
# y, x = np.where((state[:, :, 0] == 0) * (state[:, :, 1] == 0))
# state_centred = np.roll(np.roll(state, 20 - y, axis=0), 20 - x, axis=1)

# distance_map = calc_distance_map(state)
# check = np.zeros((40, 40))

# y, x = 10, 30
# for y1 in range(40):
#     for x1 in range(40):
#         check[y1, x1] = distance_map[y * 40 + x, y1 * 40 + x1]

# check[check == 1601] = 100
# plt.imshow(check)
# plt.show()
