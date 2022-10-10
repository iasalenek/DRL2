import copy
import numpy as np
from my_utils.common_utils import calc_distance_map


def compute_observation_single(state: np.ndarray,
                               id: int,
                               team: int = 0):

    y, x = np.where((state[:, :, 0] == team) * (state[:, :, 1] == id))
    state_centred = np.roll(np.roll(state, 20 - y, axis=0), 20 - x, axis=1)

    obs = np.zeros((40, 40, 3), dtype=int)
    obs[:, :, 0][state_centred[:, :, 1] == -1] = 1
    obs[:, :, 1][state_centred[:, :, 0] == 1] = 1
    obs[:, :, 2][state_centred[:, :, 0] == 0] = 1

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

            dist = distance_map[y * 40 + x, y1 * 40 + x1]

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
                10 * len(info['eaten']) + calc_closeness(next_state, distance_map))
        else:
            next_values.append(10 * len(info['eaten']))

    E_next_values = np.mean(next_values)
    #reward = (E_next_values - cur_value)
    if E_next_values != 0:
        reward = cur_value / E_next_values - 1
    else:
        reward = 0


    return reward
