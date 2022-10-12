import copy
import numpy as np
from my_utils.common_utils import calc_distance_map, calc_distance_mat


def compute_observation_single(state: np.ndarray,
                               id: int,
                               distance_map: np.ndarray,
                               team: int = 0):
    # ###
    # state = np.sum((state[:, :, 0] == team) * (state[:, :, 1] != id))
    # ###
    
    y, x = np.where((state[:, :, 0] == team) * (state[:, :, 1] == id))
    state_centred = np.roll(np.roll(state, 20 - y, axis=0), 20 - x, axis=1)

    obs = np.zeros((40, 40, 3), dtype=int)

    #Наблюдения из state

    obs[:, :, 0][state_centred[:, :, 1] == -1] = 1 # Препятствия
    obs[:, :, 1][state_centred[:, :, 0] == 1] = 1  # Жертвы
    # obs[:, :, 1][state_centred[:, :, 0] == 0] = 1  # Хищники

    # Distance map для агента
    distance_map = calc_distance_mat(distance_map, y, x)
    obs[:, :, 2] = np.roll(np.roll(distance_map, 20 - y, axis=0), 20 - x, axis=1).clip(0, 100)

    #Транспонируем наблюдения для сети
    obs = np.transpose(obs, (2, 0, 1))

    ####
    # import matplotlib.pyplot as plt
    # import time
    # plt.imshow(obs[3, :, :])
    # plt.show()
    # print(np.min(obs[3, :, :]))
    # plt.imshow(obs[2, :, :])
    # plt.show()
    # time.sleep(0)
    ####
    
    return obs


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
                     next_state: np.array,
                     info,
                     distance_map: np.ndarray,
                     n: int = 1, debug=False):

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

    # Штраф за стояние на месте
    moves = np.sum((state[:, :, 0] == 0) != (next_state[:, :, 0] == 0))

    # Считаем матожидание следущих состояний
    next_dist_n = calc_closeness_id(next_state, distance_map, ids_n)
    next_dist_n = next_dist_n[next_dist_n < 1600]
    if len(next_dist_n) > 0:
        next_value = -np.mean(next_dist_n)
        reward = next_value - cur_value + 20 * len(info['eaten']) + 5 * moves
    else:
        reward = 20 * len(info['eaten']) + 5 * moves


    # ###
    # if debug:
    #     import time
    #     if reward < -2:
    #         print('---------------')
    #         print(reward)
    #         print(dist_n)
    #         print(next_dist_n)
    #         print(len(info['eaten']))
    #         print('---------------')
    #         time.sleep(5)
    # ###

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




# Визуализация маленьких наград

# st = next_state.copy()

# check_state = np.zeros((40, 40))
# check_state[st[:, :, 0] == 0] = 2
# check_state[st[:, :, 1] == -1] = 1
# for id in ids_n:
#     check_state[st[:, :, 1] == id] = 3

# plt.imshow(check_state)
# plt.show()



#### GLOBAL CHECK!!!!

# env = OnePlayerEnv(Realm(
#         MixedMapLoader((SingleTeamLabyrinthMapLoader(),
#         # SingleTeamRocksMapLoader()
#         )),
#         1
#     ))

# done = False
# state, info = env.reset()
# step = 0
# while not done:
#     action = [np.random.randint(5) for i in range(5)]
#     next_state, done, next_info = env.step(action)
#     step += 1

# for id in range(100):
#     x0, y0, x1, y1 = None, None, None, None
#     for prey in info['preys']:
#         if prey['id'] == id:
#             x0, y0 = np.where((state[:, :, 0] == 1) * (state[:, :, 1] == id))
#     for prey in next_info['preys']:
#         if prey['id'] == id:
#             x1, y1 =np.where((next_state[:, :, 0] == 1) * (next_state[:, :, 1] == id))
        
#     if (x0 is not None) and (x1 is not None):
#         dx = (np.abs(x1 - x0)) % 39
#         dy = (np.abs(y1 - y0)) % 39
#         if (dx + dy > 1) and step != 301:
#             print(f'step: {step}   id: {id}')
#             print(f'x0: {x0}, x1: {x1}\ny0: {y0}, y1: {y1}')


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

# for y1 in range(40):
#     for x1 in range(40):
#         check[y1, x1] = distance_map[y * 40 + x, y1 * 40 + x1]

# check[check == 1601] = 100
# plt.imshow(check)
# plt.show()
