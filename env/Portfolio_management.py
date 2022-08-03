from universal.algos import *
import gym
from universal import tools


class CustomEnv(gym.Env):

    current_state = 0

    def __init__(self, df_original,df):
        super(CustomEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(15)
        # get the observation space from the data min and max
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10, 16),dtype=np.float32)
        self.df_original = df_original
        self.df = df
    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_state = 10
        return self.give_next_observation()

    def _next_observation(self):

        obs = self.df_original.iloc[self.current_state : self.current_state + 10, :]
        return obs

    def give_next_observation(self):

        obs = self.df.iloc[self.current_state : self.current_state + 10, :]
        return obs

    def _pre_observation(self):

        obs = self.df_original.iloc[self.current_state-10 : self.current_state, :]
        return obs

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        if action == 3 or action == 14:

            weights = self.result.B
            algo = BCRP()
            self.result = algo.run(self._next_observation().reset_index(drop=True))
            map_X = self.result.X
            r = (map_X - 1) * weights
            r = r.sum(axis=1) + 1
            reward = (r.prod() - 1)
            reward = reward * 20

        else:
            reward = (self.result.total_wealth - 1)
            reward = reward * 20

        self.current_state = self.current_state + 10
        done = self.current_state >= self.df.shape[0] - 20
        obs = self.give_next_observation()  # next state

        return obs, reward, done, {}, self.result

    def _take_action(self, action):

        action_type = action
        action_space = self._next_observation()

        if action_type == 1 :
            algo = BAH()
            self.result = algo.run(action_space)
        if action_type == 2:
            algo = CRP()
            self.result = algo.run(action_space)
        if action_type == 3:
            algo = BCRP()
            self.result = algo.run(self._pre_observation().reset_index(drop=True))
        if action_type == 4:
            algo = DynamicCRP()
            self.result = algo.run(action_space)
        if action_type == 6:
            algo = EG()
            self.result = algo.run(action_space)
        if action_type == 7:
            algo = Anticor()
            self.result = algo.run(action_space)
        if action_type == 8:
            algo = OLMAR()
            self.result = algo.run(action_space)
        if action_type == 9:
            algo = RMR()
            self.result = algo.run(action_space)
        if action_type == 10:
            algo = CWMR()
            self.result = algo.run(action_space)
        if action_type == 11:
            algo = WMAMR()
            self.result = algo.run(action_space)
        if action_type == 12:
            algo = BNN()
            self.result = algo.run(action_space)
        if action_type == 13:
            algo = CORN()
            self.result = algo.run(action_space)
        if action_type == 14:
            algo = BestMarkowitz()
            self.result = algo.run(self._pre_observation().reset_index(drop=True))
        if action_type == 5:
            algo = ONS()
            self.result = algo.run(action_space)
        if action_type == 0:
            algo = PAMR()
            self.result = algo.run(action_space)

    def render(self):
        print(f'Step: {self.current_state}')  # see current state
        print(f'current annualized return : {self.result.annualized_return*100}%')
        print("\n")









