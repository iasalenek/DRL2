class Agent:
    def get_actions(self, state, info):
        # TODO: implement
        return [0, 0, 0, 0, 0]

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