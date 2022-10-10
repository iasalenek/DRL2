import numpy as np
import torch
from world.my_rewards import compute_observation

DEVICE = 'cpu'

class Agent():

    def __init__(self, num_predators: int):
        self.model = torch.load("agent.pkl", map_location='cpu')
        self.num_predators = num_predators

    def get_actions(self, state, info):
        obs = compute_observation(state, self.num_predators)
        obs = np.expand_dims(obs, axis=0)
        action = self.model(torch.Tensor(obs).to(DEVICE)).view(self.num_predators, 5).argmax(axis = 1)
        return action.to(int).cpu().detach().tolist()
        #return [0, 0, 0, 0, 0]

    def reset(self, state, info):
        # TODO: implement
        pass


# class FirstTryAgent(ScriptedAgent):
#     def __init__(self, agent_pkl = None):

#         self.model = torch.load(agent_pkl).to(DEVICE)

#     def get_actions(self, state, team = None):

#         # Центрируем состояние для каждого агента
#         states = []

#         for i in range(num_predators):
#             y, x = np.where((state[:, :, 0] == 0) * (state[:, :, 1] == i))
#             states.append(np.roll(np.roll(state, 20 - y, axis=0), 20 - x, axis=1))

#         obs = torch.Tensor(np.dstack(states)).to(DEVICE).permute(2, 0, 1)
#         obs = obs.reshape(1, 10, 40, 40).to(DEVICE)
#         actions = model(obs).view(5, 5).argmax(axis=1).cpu().detach().numpy()

#         return list(actions)