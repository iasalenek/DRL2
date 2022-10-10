from world.realm import Realm
from world.envs import OnePlayerEnv, VersusBotEnv, TwoPlayerEnv
from world.utils import RenderedEnvWrapper
from world.map_loaders.base import MixedMapLoader
from world.map_loaders.single_team import SingleTeamLabyrinthMapLoader, SingleTeamRocksMapLoader
from world.map_loaders.two_teams import TwoTeamLabyrinthMapLoader, TwoTeamRocksMapLoader
from world.scripted_agents import ClosestTargetAgent


from agent import Agent


if __name__ == "__main__":
    # Task 1
    env = OnePlayerEnv(Realm(
        MixedMapLoader((SingleTeamLabyrinthMapLoader(), 
        # SingleTeamRocksMapLoader()
        )),
        1
    ))
    env = RenderedEnvWrapper(env)
    agent = Agent()
    for i in range(4):
        state, info = env.reset()
        agent.reset(state, 0)
        done = False
        while not done:
            action = agent.get_actions(state, 0)
            state, done, info = env.step(action)
        env.render(f"render/one_player_env/{i}")

    # Task 2
    # env = VersusBotEnv(Realm(
    #     MixedMapLoader((TwoTeamLabyrinthMapLoader(), TwoTeamRocksMapLoader())),
    #     2,
    #     bots={1: ClosestTargetAgent()}
    # ))
    # env = RenderedEnvWrapper(env)
    # agent = ClosestTargetAgent()
    # for i in range(4):
    #     state, info = env.reset()
    #     agent.reset(state, 0)
    #     done = False
    #     while not done:
    #         state, done, info = env.step(agent.get_actions(state, 0))
    #     env.render(f"render/versus_bot_env/{i}")

    # # Task 3
    # env = TwoPlayerEnv(Realm(
    #     MixedMapLoader((TwoTeamLabyrinthMapLoader(), TwoTeamRocksMapLoader())),
    #     2
    # ))
    # env = RenderedEnvWrapper(env)
    # agent1 = ClosestTargetAgent()
    # agent2 = ClosestTargetAgent()
    # for i in range(4):
    #     (state1, info1), (state2, info2) = env.reset()
    #     agent1.reset(state1, 0)
    #     agent2.reset(state2, 0)
    #     done = False
    #     while not done:
    #         (state1, done, info1), (state2, _, info2) = env.step(agent1.get_actions(state1, 0),
    #                                                              agent2.get_actions(state2, 0))
    #     env.render(f"render/two_player_env/{i}")
