import copy
import numpy as np

def compute_observation(state: np.ndarray, 
                        num_predators: int = 5,
                        team: int = 0):

    states = []

    for i in range(num_predators):
        y, x = np.where((state[:, :, 0] == team) * (state[:, :, 1] == i))
        states.append(np.roll(np.roll(state[:, :, 0:1], 20 - y, axis=0), 20 - x, axis=1))

    obs = np.transpose(np.dstack(states), (2, 0, 1))

    return obs

def calc_distance_map(initial_state):
    
    mask = np.zeros(initial_state.shape[:2], np.bool)
    mask[np.logical_or(np.logical_and(initial_state[:, :, 0] == -1, initial_state[:, :, 1] == 0),
                        initial_state[:, :, 0] >= 0)] = True
    mask = mask.reshape(-1)

    coords_amount = initial_state.shape[0] * initial_state.shape[1]
    distance_map = (coords_amount + 1) * np.ones((coords_amount, coords_amount))
    np.fill_diagonal(distance_map, 0.)
    distance_map[np.logical_not(mask)] = (coords_amount + 1)
    distance_map[:, np.logical_not(mask)] = (coords_amount + 1)

    indexes_helper = [
        [
            x * initial_state.shape[1] + (y + 1) % initial_state.shape[1],
            x * initial_state.shape[1] + (initial_state.shape[1] + y - 1) % initial_state.shape[1],
            ((initial_state.shape[0] + x - 1) % initial_state.shape[0]) * initial_state.shape[1] + y,
            ((x + 1) % initial_state.shape[0]) * initial_state.shape[1] + y
        ]
        for x in range(initial_state.shape[0]) for y in range(initial_state.shape[1])
    ]

    updated = True
    while updated:
        old_distances = copy.deepcopy(distance_map)
        for j in range(coords_amount):
            if mask[j]:
                for i in indexes_helper[j]:
                    if mask[i]:
                        distance_map[j] = np.minimum(distance_map[j], distance_map[i] + 1)
        updated = (old_distances != distance_map).sum() > 0

    return distance_map


def calc_closeness(state: np.ndarray, 
                   distance_map: np.ndarray,
                   num_predators: int):

    preys_team = np.max(state[:, :, 0])
    num_preys = np.sum(state[:, :, 0] == preys_team)
    preys_ind = np.where(state[:, :, 0] == preys_team)

    preys_dist = np.ones(num_preys) * 1601

    for i in range(num_predators):

        y, x = np.where((state[:, :, 0] == 0) * (state[:, :, 1] == i))
        y, x = int(x), int(y)

        for j in range(num_preys):
            y1, x1 = preys_ind[0][j], preys_ind[1][j]

            dist = distance_map[y * state.shape[0] + x, y1 * state.shape[0] + x1]

            if dist < preys_dist[j]:
                preys_dist[j] = dist

    return -np.mean(preys_dist)


def get_reward_1(env,
                 state: np.ndarray, 
                 action: np.ndarray, 
                 distance_map: np.ndarray,
                 num_predators: int,
                 k: int = 1):
    
    cur_value = calc_closeness(state, distance_map, num_predators)
    next_values = []

    # Штраф за отсутствие движения
    #next_state, done, info = copy.deepcopy(env).step(action)
    #no_movement_penalty = -np.sum(next_state[state[:, :, 0] == 0][:, 0] == 0) * 10

    for i in range(k):
        if not done:
            next_values.append(
                10 * len(info['eaten']) + calc_closeness(next_state, distance_map, num_predators))
        else:
            next_values.append(0)

    E_next_values = np.mean(next_values)
    reward = (E_next_values - cur_value)

    return reward
