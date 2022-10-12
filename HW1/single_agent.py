import numpy as np
import torch
from my_utils.common_utils import calc_distance_map
from my_utils.single_dqn_utils import compute_observation_single

DEVICE = 'cpu'

class Agent():

    def __init__(self, num_predators: int=5):
        self.model = torch.load("agent.pkl", map_location='cpu')
        self.model.eval()

    def get_actions(self, state, info):

        observations = []
        for i in range(5):
            obs = compute_observation_single(state, i, self.distance_map)
            observations.append(obs)

        observations = np.array(observations)
        observations.shape
        
        actions = self.model(torch.Tensor(observations).to(DEVICE)).argmax(axis=1)
        
        ####
        print(actions)
        ####
        
        return actions.to(int).cpu().detach().tolist()

    def reset(self, state, info):
        self.distance_map = calc_distance_map(state)
