import copy
import numpy as np
import torch
from my_utils.common_utils import calc_distance_map
from my_utils.dqn_utils import compute_observation

class Agent:
    def __init__(self):
        self.model = torch.load("agent.pkl", map_location='cpu')
        self.model.eval()

    def get_actions(self, state, info):

        observations = []
        for agent_id in range(5):
            obs = compute_observation(state, agent_id, self.distance_map)
            observations.append(obs)

        observations = np.array(observations)
        
        actions = self.model(torch.Tensor(observations)).argmax(axis=1)
        
        return actions.to(int).cpu().detach().tolist()

    def reset(self, state, info):
        self.distance_map = calc_distance_map(state)