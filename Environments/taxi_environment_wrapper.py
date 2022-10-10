import random
import gym
from Environments.MultiTaxiEnv.multitaxienv.taxi_environment import TaxiEnv
from itertools import product
import numpy as np
from Environments.abstract_wrapper_env import AbstractWrapperEnv

NEW_MAP = [
    "+---------+",
    "|X: | : :X|",
    "| : | : : |",
    "| : : : : |",
    "| : : | : |",
    "|X: :F| :X|",
    "+---------+",
]
TAXI_NAME = "taxi_1"
TAXI_INIT_POS = [4, 0]
PASS_INIT_POS = [0, 0]
PASS_DEST = [4, 4]

# ========================================== DO NOT DELETE ========================================== #
# This part is for running changing environments:
# all_possible_envs = list(product([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 4], [0, 4], [0, 4], [0, 4]))
# all_possible_envs = [list(s) for s in all_possible_envs if (s[2] != s[4] or (s[3] != s[5]))]
# global env_idx
# env_idx = -1
# in the reset function:
# self.taxis_fixed_locations = [[all_possible_envs[env_idx][0], all_possible_envs[env_idx][1]]]
# self.passengers_start_fixed_locations = [[all_possible_envs[env_idx][2], all_possible_envs[env_idx][3]]]
# self.passengers_fixed_destinations = [[all_possible_envs[env_idx][4], all_possible_envs[env_idx][5]]]
# self.passengers_locations = [self.taxis_fixed_locations[0], self.passengers_fixed_destinations[0]]
# ========================================== DO NOT DELETE ========================================== #

def set_up_env_idx():
    global env_idx
    env_idx += 1


class TaxiSimpleEnv(TaxiEnv, AbstractWrapperEnv):
    def __init__(self, _=0, num_taxis: int = 1, num_passengers: int = 1, max_fuel: list = None,
                 domain_map: list = None, taxis_capacity: list = None, collision_sensitive_domain: bool = True,
                 fuel_type_list: list = None, option_to_stand_by: bool = False):
        # Initializing default values
        self.transform_num = 10
        self.num_taxis = num_taxis
        if max_fuel is None:
            self.max_fuel = [100] * num_taxis
        else:
            self.max_fuel = max_fuel

        if domain_map is None:
            self.desc = np.asarray(NEW_MAP, dtype='c')
        else:
            self.desc = np.asarray(domain_map, dtype='c')

        if taxis_capacity is None:
            self.taxis_capacity = [1] * num_passengers
        else:
            self.taxis_capacity = taxis_capacity

        if fuel_type_list is None:
            self.fuel_type_list = ['F'] * num_passengers
        else:
            self.fuel_type_list = fuel_type_list

        # Relevant features for map orientation, notice that we can only drive between the columns (':')
        self.num_rows = num_rows = len(self.desc) - 2
        self.num_columns = num_columns = len(self.desc[0][1:-1:2])

        # Set locations of passengers and fuel stations according to the map.
        self.passengers_locations = []
        self.fuel_station1 = None
        self.fuel_station2 = None
        self.fuel_stations = []

        # initializing map with passengers and fuel stations
        for i, row in enumerate(self.desc[1:-1]):
            for j, char in enumerate(row[1:-1:2]):
                loc = [i, j]
                if char == b'X':
                    self.passengers_locations.append(loc)
                elif char == b'F':
                    self.fuel_station1 = loc
                    self.fuel_stations.append(loc)
                elif char == b'G':
                    self.fuel_station2 = loc
                    self.fuel_stations.append(loc)

        self.coordinates = [[i, j] for i in range(num_rows) for j in range(num_columns)]

        # self.num_taxis = num_taxis
        self.taxis_names = ["taxi_" + str(index + 1) for index in range(self.num_taxis)]

        self.collision_sensitive_domain = collision_sensitive_domain

        # Indicator list of 1's (collided) and 0's (not-collided) of all taxis
        self.collided = np.zeros(self.num_taxis)

        self.option_to_standby = option_to_stand_by

        # A list to indicate whether the engine of taxi i is on (1) or off (0), all taxis start as on.
        self.engine_status_list = list(np.ones(self.num_taxis).astype(bool))

        self.num_passengers = num_passengers

        # Available actions in relation to all actions based on environment parameters.
        self.available_actions_indexes, self.index_action_dictionary, self.action_index_dictionary \
            = self._set_available_actions_dictionary()
        self.num_actions = len(self.available_actions_indexes)
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.MultiDiscrete(self._get_observation_space_list())
        self.bounded = False

        self.last_action = None
        self.num_states = self._get_num_states()

        self._seed()
        self.state = None
        self.dones = {taxi_name: False for taxi_name in self.taxis_names}
        self.dones['__all__'] = False

        self.np_random = None

        self.taxis_fixed_locations = [TAXI_INIT_POS]
        self.passengers_start_fixed_locations = [PASS_INIT_POS]
        self.passengers_fixed_destinations = [PASS_DEST]
        self.passengers_locations = [PASS_INIT_POS, PASS_DEST]

        self.reset()

    def reset(self):
        obs = super(TaxiSimpleEnv, self).reset()[TAXI_NAME]
        encoded_state = self.encode(obs)
        return encoded_state

    def step(self, action):
        action_dict = {TAXI_NAME: action}
        next_state, reward, dones, info = super(TaxiSimpleEnv, self).step(action_dict)
        next_state, reward, dones = next_state[TAXI_NAME][0], reward[TAXI_NAME], dones[TAXI_NAME]
        encoded_next_state = self.encode(next_state)
        return encoded_next_state, reward, dones, info

    def encode(self, state):
        # (self.num_rows), self.num_columns, max_fuel[0] + 1, self.num_rows, self.num_columns, self.passengers_locations, 4
        taxi_row, taxi_col, fuel, pass_loc_x, pass_loc_y, dest_idx_x, dest_idx_y, pass_status = state[0], state[1], \
                                                                                                state[2], state[3], \
                                                                                                state[4], state[5], \
                                                                                                state[6], state[7]
        # dest_idx = self._get_pass_dest_idx(dest_idx_x, dest_idx_y)

        i = taxi_row

        i *= self.num_columns
        i += taxi_col

        i *= (self.max_fuel[0] + 1)
        i += fuel

        i *= self.num_rows
        i += pass_loc_x

        i *= self.num_columns
        i += pass_loc_y

        # i *= len(self.passengers_fixed_destinations)
        # i += dest_idx

        i *= 3
        i += pass_status
        return i

    def decode(self, i):
        # 4, self.passengers_locations, self.num_columns, self.num_rows, max_fuel[0] + 1, self.num_columns, self.num_rows
        j = i
        out = []

        passenger_status = [(i % 3)]
        out.append(passenger_status)
        i = i // 3

        # passenger_dest_idx = [self.passengers_fixed_destinations[i % len(self.passengers_fixed_destinations)]]
        out.append([PASS_DEST])
        # i = i // len(self.passengers_fixed_destinations)

        passenger_loc_y = i % self.num_columns
        i = i // self.num_columns
        passenger_loc_x = i % self.num_rows
        i = i // self.num_rows
        passenger_location = [[passenger_loc_x, passenger_loc_y]]
        out.append(passenger_location)

        fuel = [i % (self.max_fuel[0] + 1)]
        out.append(fuel)
        i = i // (self.max_fuel[0] + 1)

        taxi_y = i % self.num_columns
        i = i // self.num_columns
        taxi_x = i
        taxi_loc = [[taxi_x, taxi_y]]
        out.append(taxi_loc)

        assert 0 <= i < self.num_rows

        return list(reversed(out))

    def _get_pass_dest_idx(self, dest_idx_x, dest_idx_y):
        dest_idx = -1
        for i, loc in enumerate(self.passengers_locations):
            if dest_idx_x == loc[0] and dest_idx_y == loc[1]:
                dest_idx = i
                break
        if dest_idx == -1:
            raise Exception("no such destination!")
        return dest_idx

    def flatten_state(self, state):
        taxi_loc, fuel, pas_loc, pas_des, status = state[0][0], state[1], state[2][0], state[3][0], state[4]
        return taxi_loc + fuel + pas_loc + pas_des + status

    def get_states_from_partial_obs(self, partial_obs):
        partial_obs_aligned_with_env = False
        iter_num = 200
        while not partial_obs_aligned_with_env and iter_num != 0:
            obs = self.reset()
            obs = self.flatten_state(self.decode(obs))
            if self._is_aligned(obs, partial_obs):
                partial_obs_aligned_with_env = True
            iter_num -= 1

        if partial_obs_aligned_with_env:
            taxi_x = [partial_obs[0]] if (partial_obs[0] is not None) else list(range(self.num_columns))
            taxi_y = [partial_obs[1]] if (partial_obs[1] is not None) else list(range(self.num_rows))
            fuel = [partial_obs[2]] if partial_obs[2] else list(range(1, self.max_fuel[0]))
            passenger_start_x, passenger_start_y = [obs[3]], [obs[4]]
            passenger_dest_x, passenger_dest_y = [obs[5]], [obs[6]]
            passenger_status = [partial_obs[7]] if partial_obs[7] else list(range(1, 4))
            states = list(
                product(taxi_x, taxi_y, fuel, passenger_start_x, passenger_start_y, passenger_dest_x, passenger_dest_y,
                        passenger_status, repeat=1))
            states = [self.encode(state) for state in states]
        else:
            states = []
        return states

    def _is_aligned(self, obs, partial_obs):
        passenger_start_x, passenger_start_y, passenger_dest_x, passenger_dest_y = self._get_passenger_info(partial_obs)
        return (passenger_start_x is None or passenger_start_x == obs[3]) and (
                passenger_start_y is None or passenger_start_y == obs[4]) and (
                       passenger_dest_x is None or passenger_dest_x == obs[5]) and (
                       passenger_dest_y is None or passenger_dest_y == obs[6]) and (
            partial_obs[7] == obs[7])

    def _get_passenger_info(self, partial_obs):
        passenger_start_x, passenger_start_y = partial_obs[3], partial_obs[4]
        passenger_dest_x, passenger_dest_y = partial_obs[5], partial_obs[6]
        return passenger_start_x, passenger_start_y, passenger_dest_x, passenger_dest_y


class TaxiSimpleExampleEnv(TaxiSimpleEnv):
    """
    This environment fixes the starting state for every episode to be:
    taxi starting location - (4,0)
    taxi fuel level - 3
    passenger starting location - (0, 0)
    passenger destination - (4, 4)
    """

    def reset(self) -> dict:
        """
        Reset the environment's state:
            - taxis coordinates - fixed.
            - refuel all taxis
            - destinations - fixed.
            - passengers locations - fixed.
            - preserve other definitions of the environment (collision, capacity...)
            - all engines turn on.
        Args:

        Returns: The reset state.

        """
        # reset taxis locations
        taxis_locations = [TAXI_INIT_POS]
        self.collided = np.zeros(self.num_taxis)
        self.bounded = False
        self.window_size = 5
        self.counter = 0

        # refuel everybody
        fuels = [3 for i in range(self.num_taxis)]

        # reset passengers
        passengers_start_location = self.passengers_start_fixed_locations
        passengers_destinations = self.passengers_fixed_destinations
        self.passengers_locations = [passengers_start_location[0], passengers_destinations[0]]

        # Status of each passenger: delivered (1), in_taxi (positive number>2), waiting (2)
        passengers_status = [2 for _ in range(self.num_passengers)]
        self.state = [taxis_locations, fuels, passengers_start_location, passengers_destinations, passengers_status]

        self.last_action = None
        # Turning all engines on
        self.engine_status_list = list(np.ones(self.num_taxis))

        # resetting dones
        self.dones = {taxi_id: False for taxi_id in self.taxis_names}
        self.dones['__all__'] = False
        obs = {}
        for taxi_id in self.taxis_names:
            obs[taxi_id] = self.get_observation(self.state, taxi_id)
        obs = obs[TAXI_NAME][0]
        encoded_state = self.encode(obs)
        decoded_state = self.decode(encoded_state)
        return encoded_state

