import numpy as np
import torch
from my_utils import compute_observation

import matplotlib.pyplot as plt
import time

DEVICE = 'cpu'

class Agent():

    def __init__(self, num_predators: int=5):
        self.model = torch.load("agent.pkl", map_location='cpu')
        self.num_predators = num_predators

    def get_actions(self, state, info):
        obs = compute_observation(state, self.num_predators)
        obs = np.expand_dims(obs, axis=0)

        action = self.model(torch.Tensor(obs).to(DEVICE)).view(self.num_predators, 5).argmax(axis = 1)

        action = action.to(int).cpu().detach().tolist()
        print(action)
        #print(self.model(torch.Tensor(obs).to(DEVICE)))
        #time.sleep(5)
        return action
        #return [0, 0, 0, 0, 0]

    def reset(self, state, info):
        # TODO: implement
        pass


# Test


# num_predators = 5
# obs = compute_observation(state, num_predators)
# obs = np.expand_dims(obs, axis=0)

# model(torch.Tensor(obs).to('cpu')).view(num_predators, 5)

# action = model(torch.Tensor(obs).to('cpu')).view(num_predators, 5).argmax(axis = 1)
