from itertools import product

from Environments.SingleTaxiEnv.single_small_taxi_env import *
from Environments.abstract_wrapper_env import AbstractWrapperEnv


class SingleTaxiSimpleEnv(SingleTaxiEnv, AbstractWrapperEnv):
    def __init__(self, deterministic=True):
        super().__init__(deterministic)
        self.transform_num = 7

    def is_possible_initial_state(self, pass_idx, dest_idx, row, col):  # Fix taxi initial position
        return pass_idx < 4 and pass_idx != dest_idx and row == self.init_row and col == self.init_col

    def get_states_from_partial_obs(self, partial_obs):
        partial_obs_aligned_with_env = False
        iter_num = 200
        while not partial_obs_aligned_with_env and iter_num != 0:
            obs = self.decode(self.reset())
            if self._is_aligned(obs, partial_obs):
                partial_obs_aligned_with_env = True
            iter_num -= 1

        if partial_obs_aligned_with_env:
            taxi_x = [partial_obs[0]] if (partial_obs[0] is not None) else list(range(self.num_columns))
            taxi_y = [partial_obs[1]] if (partial_obs[1] is not None) else list(range(self.num_rows))
            passenger_loc = [partial_obs[2]] if (partial_obs[2] is not None) else list(
                range(len(self.passengers_locations)))
            passenger_dest = [partial_obs[3]] if (partial_obs[3] is not None) else list(
                range(len(self.passengers_locations)))
            fuel = [partial_obs[4]] if partial_obs[4] else list(range(0, MAX_FUEL))
            states = list(product(taxi_x, taxi_y, passenger_loc, passenger_dest, fuel, repeat=1))
            states = [self.encode(*state) for state in states]
        else:
            states = []
        return states

    def _is_aligned(self, obs, partial_obs):
        #  taxi_x, taxi_y, passenger_loc, passenger_dest, fuel
        taxi_x, taxi_y, passenger_loc, passenger_dest, fuel = obs
        return passenger_dest == partial_obs[3]
