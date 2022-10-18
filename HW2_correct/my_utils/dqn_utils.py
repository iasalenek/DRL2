import copy
import numpy as np
from my_utils.common_utils import calc_distance_map, calc_distance_mat


def compute_observation(state: np.ndarray,
                        id: int,
                        distance_map: np.ndarray,
                        team: int = 0):

    # preys_team = np.max(state[:, :, 0])

    y, x = np.where((state[:, :, 0] == team) * (state[:, :, 1] == id))
    state_centred = np.roll(np.roll(state, 20 - y, axis=0), 20 - x, axis=1)

    obs = np.zeros((40, 40, 5), dtype=int)

    #Наблюдения из state

    obs[:, :, 0][state_centred[:, :, 1] == -1] = 1 # Препятствия
    obs[:, :, 1][state_centred[:, :, 0] == 2] = 1  # Жертвы
    obs[:, :, 2][state_centred[:, :, 0] == 0] = 1  # Хищники
    obs[:, :, 3][state_centred[:, :, 0] == 1] = 1  # Враги

    # Distance map для агента
    distance_map = calc_distance_mat(distance_map, y, x)
    obs[:, :, 4] = np.roll(np.roll(distance_map, 20 - y, axis=0), 20 - x, axis=1)# .clip(0, 100)
    obs[:, :, 4][obs[:, :, 3] > 1600] = 0

    #Транспонируем наблюдения для сети
    obs = np.transpose(obs, (2, 0, 1))

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


def closest_n_reward(id: int,
                     state: np.ndarray,
                     next_state: np.array,
                     info,
                     distance_map: np.ndarray,
                     n: int = 5, debug=False):

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
    moves_id = np.sum(
        (state[:, :, 0] == 0) * (state[:, :, 1] == id) != 
        (next_state[:, :, 0] == 0) * (next_state[:, :, 1] == id)) // 2

    eaten_id = np.sum([v[1] == id for k, v in info['eaten'].items()])

    next_dist_n = calc_closeness_id(next_state, distance_map, ids_n)
    next_dist_n = next_dist_n[next_dist_n < 1600]


    if len(next_dist_n) > 0:
        next_value = -np.mean(next_dist_n)
        reward = next_value - cur_value + 20 * eaten_id + 5 * moves_id
    else:
        reward = 20 * eaten_id + 5 * moves_id

    return reward
