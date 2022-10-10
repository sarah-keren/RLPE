import gym
import numpy as np
from queue import Queue

# produce a shortest path tree for the shortest ways to get to the location specified
# loc: tuple (row, col) of desired location  
# desc: description of environment map (env.desc)
from Observer.abstract_expert import AbstractExpert


def shortest_path_tree(desc, loc):
    (loc_r, loc_c) = loc
    num_rows = len(desc) - 2
    num_cols = len(desc[0][1:-1:2])
    max_row = num_rows - 1
    max_col = num_cols - 1
    spt = [[[[], np.inf] for _ in range(num_cols)] for _ in range(num_rows)]

    spt[loc_r][loc_c][0].append(-1)  # after arriving at dest, no action should be taken
    spt[loc_r][loc_c][1] = 0

    q = Queue()
    q.put([loc, 0])
    while not q.empty():
        [(row, col), cur_dist] = q.get()  # current location we are exploring the neighbors of
        for action in range(4):
            if action == 0:
                (next_r, next_c) = (min(row + 1, max_row), col)
                if spt[next_r][next_c][1] >= cur_dist + 1:
                    if spt[next_r][next_c][1] > cur_dist + 1:
                        spt[next_r][next_c][1] = cur_dist + 1
                        q.put([(next_r, next_c), cur_dist + 1])
                    spt[next_r][next_c][0].append(1)  # add to optimal actions list

            elif action == 1:
                (next_r, next_c) = (max(row - 1, 0), col)
                if spt[next_r][next_c][1] >= cur_dist + 1:
                    if spt[next_r][next_c][1] > cur_dist + 1:
                        spt[next_r][next_c][1] = cur_dist + 1
                        q.put([(next_r, next_c), cur_dist + 1])
                    spt[next_r][next_c][0].append(0)  # add to optimal actions list

            elif action == 2 and desc[1 + row, 2 * col + 2] == b":":
                (next_r, next_c) = (row, min(col + 1, max_col))

                if spt[next_r][next_c][1] >= cur_dist + 1:
                    if spt[next_r][next_c][1] > cur_dist + 1:
                        spt[next_r][next_c][1] = cur_dist + 1
                        q.put([(next_r, next_c), cur_dist + 1])
                    spt[next_r][next_c][0].append(3)  # add to optimal actions list

            elif action == 3 and desc[1 + row, 2 * col] == b":":
                (next_r, next_c) = (row, max(col - 1, 0))

                if spt[next_r][next_c][1] >= cur_dist + 1:
                    if spt[next_r][next_c][1] > cur_dist + 1:
                        spt[next_r][next_c][1] = cur_dist + 1
                        q.put([(next_r, next_c), cur_dist + 1])
                    spt[next_r][next_c][0].append(2)  # add to optimal actions list

    return spt


# Hard-coded expert policy for Taxi-v2 domain
# Finds a policy set: a list of actions that would all count as part of the optimal policy

# Taxi Actions:
# There are 6 discrete deterministic actions:
# - 0: move south
# - 1: move north
# - 2: move east 
# - 3: move west 
# - 4: pickup passenger
# - 5: dropoff passenger


class Taxi_Expert(AbstractExpert):
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
        # self.ignored_taxi_locations = [(0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4)]
        self.ignored_taxi_locations = []

    # get expert action, given a tuple of the form:
    # (taxi_loc_x, taxi_loc_y, fuel_level, pass_start_x, pass_start_y, pass_dest_x, pass_dest_y, pass_status)
    # unnecessary information in the tuple can be left as None (e.g. fuel level is not used in this function and can be None)
    def get_expert_policy_set(self, state_tuple):
        (taxi_loc_x, taxi_loc_y, fuel_level, pass_start_x, pass_start_y, pass_dest_x, pass_dest_y,
         pass_status) = state_tuple
        taxi_loc = (taxi_loc_x, taxi_loc_y)
        pass_start = (pass_start_x, pass_start_y)
        pass_dest = (pass_dest_x, pass_dest_y)
        if pass_status == 3 and taxi_loc == pass_dest:
            return [5]  # dropoff
        elif pass_status == 3 and taxi_loc != pass_dest:
            return self.shortest_path_trees[pass_dest][taxi_loc_x][taxi_loc_y][0]
        elif pass_status == 2 and taxi_loc == pass_start:
            return [4]  # pickup
        elif pass_status == 2 and taxi_loc != pass_start:
            return self.shortest_path_trees[pass_start][taxi_loc_x][taxi_loc_y][0]
        elif pass_status == 1:
            print("Passenger has already arrived")
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
                for pass_start in self.env.passengers_start_fixed_locations:
                    for pass_dest in self.env.passengers_fixed_destinations:
                        for pass_status in [2, 3]:
                            state_tuple = (
                                taxi_loc_x, taxi_loc_y, fuel_level, pass_start[0], pass_start[1], pass_dest[0],
                                pass_dest[1], pass_status)
                            expert_policy_dict[state_tuple] = self.get_expert_policy_set(state_tuple)
        return expert_policy_dict


