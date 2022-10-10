import copy
import numpy as np


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
