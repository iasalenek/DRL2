import copy
import numpy as np
from my_utils.common_utils import calc_distance_map, calc_distance_mat


def compute_observation(id: int,
                        state: np.ndarray,
                        distance_map: np.ndarray,
                        agent_team: int = 0,
                        enemy_team: int = 1,
                        preys_team: int = 2):

    y, x = np.where((state[:, :, 0] == agent_team) * (state[:, :, 1] == id))
    state_centred = np.roll(np.roll(state, 20 - y, axis=0), 20 - x, axis=1)

    #Наблюдения из state
    tiles = (state_centred[:, :, 1] == -1).astype(int) # Препятствия
    agents = (state_centred[:, :, 0] == agent_team).astype(int)  # Хищники
    enemies = (state_centred[:, :, 0] == enemy_team).astype(int)  # Враги
    preys = (state_centred[:, :, 0] == preys_team).astype(int)  # Жертвы

    # Distance map для агента
    distance_mat = calc_distance_mat(distance_map, y, x)
    distance_mat = np.roll(np.roll(distance_mat, 20 - y, axis=0), 20 - x, axis=1)

    # Собираем все наблюдения в одно
    obs = np.dstack([
        tiles,
        agents,
        enemies,
        preys,
        distance_mat
    ])

    #Транспонируем наблюдения для сети
    obs = np.transpose(obs, (2, 0, 1))

    return obs


#### Test
# import matplotlib.pyplot as plt

# env = TwoPlayerEnv(Realm(
#         MixedMapLoader((TwoTeamLabyrinthMapLoader(), TwoTeamRocksMapLoader())),
#         2
#     ))

# state_info_0, state_info_1 = env.reset()

# state, info = env.reset()[0]
# distance_map = calc_distance_map(state)
# next_state, done, info = env.step([1, 2, 3, 4, 0], [0, 1, 2, 3, 4])[0]


# done = False
# while not done:
#     next_state, done, info = env.step([0, 0, 0, 0, 0])
#     if len(info['eaten']) > 0:
#         break

# info['eaten']

# obs = compute_observation(0, state, env, distance_map)
# obs[[3, 4]].shape
# plt.imshow(obs[3] + 2 * obs[4])
# plt.imshow(obs[5])
# plt.show()
# ####

# state_0[:, :, 1].isin([0, 1, 2])

# plt.imshow(np.isin(state_0[:, :, 1], [1, 2, 3]))
# plt.show()

# calc_closeness(state, 
#                agent_ids = [0], 
#                target_ids = [0, 1, 2],
#                target_team = 0, 
#                distance_map=distance_map)

# plt.imshow((state[:, :, 0] == -1) * (state[:, :, 1] == 0) + \
#            (state[:, :, 0] == 2) * np.isin(state[:, :, 1], [33]) * 2 + \
#            (state[:, :, 0] == 0) * np.isin(state[:, :, 1], [0]) * 3)
# plt.show()


def calc_closeness(state: np.ndarray,
                   agent_ids: np.ndarray,
                   target_ids: np.ndarray,
                   distance_map: np.ndarray,
                   agent_team: int = 0,
                   target_team: int=2):

    # Переводим все в np array
    agent_ids = np.array(agent_ids, ndmin=1)
    target_ids = np.array(target_ids, ndmin=1)

    if agent_ids is None:
        agent_ids = np.arange(5)

    agent_ind = np.where(
        (state[:, :, 0] == agent_team) * np.isin(
         state[:, :, 1], agent_ids))

    target_ind = np.where(
        (state[:, :, 0] == target_team) * np.isin(
         state[:, :, 1], target_ids))

    target_dist = np.ones_like(target_ids,) * 1601

    for y_t, x_t in zip(*target_ind):

        target_id = state[y_t, x_t][1]

        for y_a, x_a in zip(*agent_ind):

            dist = distance_map[y_a * 40 + x_a, y_t * 40 + x_t]

            if dist < target_dist[np.where(target_ids == target_id)]:
                target_dist[np.where(target_ids == target_id)] = dist

    return target_dist


def vs_agent_reward(state: np.ndarray,
                    agent_id: int,
                    next_state: np.array,
                    info,
                    distance_map: np.ndarray):

    eat = False  # жертву съел агент
    loss = False # жертву съели перед агентом
    was_eaten = False # съели агента
    closest_prey_distance_reduction = 0 # снижение расстояния до ближайшей жертвы
    moved = False # агент сделал шаг

    # Расстояние до ближайшей жертвы в начальном состоянии
    preys_left_ids = state[:, :, 1][state[:, :, 0] == 2]

    preys_left_dist = calc_closeness(
        state, agent_id, preys_left_ids, distance_map)

    closest_prey_dist = np.min(preys_left_dist)
    closest_prey_id = preys_left_ids[np.argmin(preys_left_dist)]

    if closest_prey_dist > 1000:
        raise ValueError('closest_prey_dist > 1000???')

    # Расстояние до той же жертвы в следующем состоянии
    next_closest_prey_dist = calc_closeness(
        next_state, agent_id, closest_prey_id, distance_map)[0]

    # Если ближайшую жертву съели, то проверяем кто
    if next_closest_prey_dist == 1601:
        # Съел агент
        eat = (0, agent_id) in info['eaten'].values()
        # Cъел не агент
        loss = not eat
    # Если ближайшую жертву не съели, то считаем снижение расстояния до нее
    else:
        closest_prey_distance_reduction = \
            closest_prey_dist - next_closest_prey_dist

    # Штраф за отсутствие движения
    y, x = np.where((state[:, :, 0] == 0) * (state[:, :, 1] == agent_id))
    if (state[y, x] != next_state[y, x]).any():
        moved = True

    # Штраф если агента съели
    was_eaten = (0, id) in info['eaten'].keys()
        
    reward = (10 * eat
             -10 * loss
             -20 * was_eaten
             + 2 * closest_prey_distance_reduction
             + 1 * moved)

    return reward
