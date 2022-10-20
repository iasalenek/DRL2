import copy
import numpy as np
from my_utils.common_utils import calc_distance_map, calc_distance_mat
from world.scripted_agents import BrokenClosestTargetAgent


def next_bot_step(state, env, bot_team: int = 1):

    new_position = np.zeros((40, 40), dtype=int)

    realm = env.realm
    world = realm.world

    actions = realm.bots[bot_team].get_actions(state, bot_team)

    for i in range(5):
        ent = world.teams[bot_team][i]
        act = actions[i]

        _, _, tx, ty = world._action_coord_change(ent, act)

        new_position[ty, tx] = 1

    return new_position
        

def compute_observation(id: int,
                        state: np.ndarray,
                        bots_next: np.ndarray,
                        distance_map: np.ndarray,
                        agent_team: int = 0,
                        bot_team: int = 1):

    y, x = np.where((state[:, :, 0] == agent_team) * (state[:, :, 1] == id))
    state_centred = np.roll(np.roll(state, 20 - y, axis=0), 20 - x, axis=1)

    # obs = np.zeros((40, 40, 5), dtype=int)

    #Наблюдения из state
    tiles = (state_centred[:, :, 1] == -1).astype(int) # Препятствия
    preys = (state_centred[:, :, 0] == 2).astype(int)  # Жертвы
    agents = (state_centred[:, :, 0] == 0).astype(int)  # Хищники
    bots_cur = (state_centred[:, :, 0] == 1).astype(int)  # Враги

    # Distance map для агента
    distance_mat = calc_distance_mat(distance_map, y, x)
    distance_mat = np.roll(np.roll(distance_mat, 20 - y, axis=0), 20 - x, axis=1)

    # Центрируем следующее состояние бота
    bots_next = np.roll(np.roll(bots_next, 20 - y, axis=0), 20 - x, axis=1)

    # Собираем все наблюдения в одно
    obs = np.dstack([
        tiles,
        preys,
        agents,
        bots_cur,
        bots_next,
        distance_mat
    ])

    #Транспонируем наблюдения для сети
    obs = np.transpose(obs, (2, 0, 1))

    return obs


# #### Test
# import matplotlib.pyplot as plt

# env = VersusBotEnv(Realm(
#         MixedMapLoader((TwoTeamLabyrinthMapLoader(), TwoTeamRocksMapLoader())),
#         2,
#         bots={1: ClosestTargetAgent()}
#     ))

# state, info = env.reset()

# done = False
# while not done:
#     next_state, done, info = env.step([0, 0, 0, 0, 0])
#     if len(info['eaten']) > 0:
#         break

# info['eaten']

# distance_map = calc_distance_map(state)

# obs = compute_observation(0, state, env, distance_map)
# obs[[3, 4]].shape
# plt.imshow(obs[3] + 2 * obs[4])
# plt.imshow(obs[5])
# plt.show()
# ####


def calc_closeness_id(state: np.ndarray,
                      distance_map: np.ndarray,
                      ids: np.ndarray,
                      agent_id: None,
                      team: int=0):

    # Здесь команда охотников равна 0

    preys_team = np.max(state[:, :, 0])
    preys_ind = np.where(state[:, :, 0] == preys_team)
    if agent_id is not None:
        predators_ind = np.where(
            (state[:, :, 0] == team) * (state[:, :, 1] == agent_id))
    else:
        predators_ind = np.where(state[:, :, 0] == team)

    preys_dist = np.ones_like(ids) * 1601

    for y1, x1 in zip(*preys_ind):

        id = state[y1, x1][1]
        if id in ids:

            for y, x in zip(*predators_ind):
                dist = distance_map[y * 40 + x, y1 * 40 + x1]

                if dist < preys_dist[np.where(ids == id)]:
                    preys_dist[np.where(ids == id)] = dist

    return preys_dist


def vs_bot_reward(id: int,
                  state: np.ndarray,
                  next_state: np.array,
                  info,
                  distance_map: np.ndarray,
                  n: int = 5, debug=False):
    pass


def closest_n_reward(id: int,
                     state: np.ndarray,
                     next_state: np.array,
                     info,
                     distance_map: np.ndarray,
                     n: int = 1, debug=False):

    # Все 100 жертв
    all_ids = np.arange(100)
    all_dist = calc_closeness_id(state, distance_map, all_ids, id, team=0)

    # Только оставшиеся и достижимые жертвы
    ids = all_ids[all_dist < 1600]
    dist = all_dist[all_dist < 1600]

    # Расстояния до этих жертв у бота
    bot_dist = calc_closeness_id(state, distance_map, ids, None, team=1)
    ids = ids[dist < bot_dist]
    dist = dist[dist < bot_dist]

    closeness = 0

    if len(dist) > 0:
        

        closest_id = np.array([ids[np.argmin(dist)]])
        cur_closest_dist = np.min(dist)
        next_closest_dist = calc_closeness_id(next_state, distance_map, closest_id, id, team=0)[0]

        if not np.isnan(next_closest_dist) and next_closest_dist != 1601:
            closeness = cur_closest_dist - next_closest_dist

            # ###
            # import time
            # print(closeness)
            # time.sleep(0.2)
            # ###

    # Штраф за стояние на месте
    moves_id = np.sum(
        (state[:, :, 0] == 0) * (state[:, :, 1] == id) != 
        (next_state[:, :, 0] == 0) * (next_state[:, :, 1] == id)) // 2

    # Cъел или был съеден
    eat = (0, id) in info['eaten'].values()
    was_eaten = (0, id) in info['eaten'].keys()

    # Delta score
    # delta_score = next_info['scores'][0] - info['scores'][0]

    reward = 5 * closeness + 5 * moves_id + 20 * eat - 60 * was_eaten

    return reward
