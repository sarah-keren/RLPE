from Observer.abstract_expert import AbstractExpert
from Environments.SingleTaxiEnv.single_small_taxi_env import *
from Observer.taxi_expert import shortest_path_tree


class SingleTaxiExpert(AbstractExpert):
    def __init__(self, env):
        self.env = env
        self.desc = env.desc  # array holding map layout
        self.locs = env.passengers_locations  # locations of passenger pickup or dropoff X's on map
        self.num_rows = len(self.desc) - 2
        self.num_cols = len(self.desc[0][1:-1:2])
        self.max_row = self.num_rows - 1
        self.max_col = self.num_cols - 1
        # self.shortest_path_trees = []
        # for loc in self.locs:
        #     self.shortest_path_trees.append(shortest_path_tree(self.desc, loc))
        self.shortest_path_trees = {}
        for loc in env.passengers_locations:
            self.shortest_path_trees[tuple(loc)] = shortest_path_tree(self.desc, loc)
        self.ignored_taxi_locations = [(0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4)]

    # get expert action, given a tuple of the form:
    # (taxi_loc_x, taxi_loc_y, fuel_level, pass_start_x, pass_start_y, pass_dest_x, pass_dest_y, pass_status)
    # unnecessary information in the tuple can be left as None
    # (e.g. fuel level is not used in this function and can be None)
    def get_expert_policy_set(self, state_tuple):
        (taxi_loc_x, taxi_loc_y, pass_loc_idx, pass_dest_idx, fuel) = state_tuple
        taxi_loc = (taxi_loc_x, taxi_loc_y)
        pass_loc = self.env.passengers_locations[pass_loc_idx] if pass_loc_idx != self.env.passenger_in_taxi else None
        pass_dest = self.env.passengers_locations[pass_dest_idx]
        if pass_loc_idx == self.env.passenger_in_taxi and taxi_loc == pass_dest:
            return [DROPOFF]  # dropoff
        elif pass_loc_idx == self.env.passenger_in_taxi and taxi_loc != pass_dest:
            return self.shortest_path_trees[pass_dest][taxi_loc_x][taxi_loc_y][0]
        elif pass_loc_idx != self.env.passenger_in_taxi and taxi_loc == pass_loc and pass_loc_idx != pass_dest_idx:
            return [PICKUP]  # pickup
        elif pass_loc_idx != self.env.passenger_in_taxi and taxi_loc != pass_loc and pass_loc_idx != pass_dest_idx:
            return self.shortest_path_trees[pass_loc][taxi_loc_x][taxi_loc_y][0]
        elif pass_loc_idx != self.env.passenger_in_taxi and pass_loc_idx == pass_dest_idx:
            # print("Passenger has already arrived")
            return None
        else:
            print("Not a valid state")

    def full_expert_policy_dict(self):
        fuel_level = None
        expert_policy_dict = {}
        for taxi_loc_x in range(self.num_rows):
            for taxi_loc_y in range(self.num_cols):
                if (taxi_loc_x, taxi_loc_y) in self.ignored_taxi_locations:
                    continue
                for pass_start_idx in range(len(self.env.passengers_locations)):
                    for pass_dest_idx in range(len(self.env.passengers_locations)):
                        state_tuple = (taxi_loc_x, taxi_loc_y, pass_start_idx, pass_dest_idx, fuel_level)
                        optimal_action = self.get_expert_policy_set(state_tuple)
                        if optimal_action:
                            expert_policy_dict[state_tuple] = optimal_action
        return expert_policy_dict
