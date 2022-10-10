import numpy as np
import torch
from my_utils.single_dqn_utils import compute_observation_single

DEVICE = 'cpu'

class Agent():

    def __init__(self, num_predators: int=5):
        self.model = torch.load("agent.pkl", map_location='cpu')

    def get_actions(self, state, info):

        observations = []
        for i in range(5):
            obs = compute_observation_single(state, id=i)
            observations.append(obs)

        observations = np.array(observations)
        observations.shape
        
        actions = self.model(torch.Tensor(observations).to(DEVICE)).argmax(axis=1)
        return actions.to(int).cpu().detach().tolist()

    def reset(self, state, info):
        # TODO: implement
        pass
