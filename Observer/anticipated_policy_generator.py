from constants import *
from Agents.RL_agents import rl_agent
import numpy as np


# helper function for flattening irregular nested tuples
def mixed_flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__"):
            result.extend(mixed_flatten(el))
        else:
            result.append(el)
    return result


# helper function for making a list of coordinates of interest
# make list of valid coords in environment within dist of the given loc
def get_nearby_coords(env, loc, dist):  # option for later: add an option to change definition of distance
    max_rows = env.num_rows - 1
    max_cols = env.num_columns - 1
    (x, y) = loc
    result = []
    for i in range(x - dist, x + dist + 1):
        for j in range(y - dist, y + dist + 1):
            if 0 <= i <= max_rows and 0 <= j <= max_cols:
                result.append((i, j))
    return result


def sample_anticipated_policy(policy_dict, num_states_in_partial_policy):
    # get flat indices of sampled states
    sampled_states_flat = np.random.choice(len(policy_dict), size=num_states_in_partial_policy, replace=False)

    partial_sampled_policy = {}
    for i, item in enumerate(policy_dict.items()):
        if i in sampled_states_flat:
            partial_sampled_policy[tuple(item[0])] = item[1]
    return partial_sampled_policy


def is_interesting_state(state, passenger_origins, passenger_destinations):
    taxi_location = [state[0], state[1]]
    fuel_level = state[2]
    passenger_location = [state[3], state[4]]
    passenger_destination = [state[5], state[6]]
    passenger_status = state[6]

    fuel_is_full = (fuel_level == 100) or (not fuel_level)
    taxi_in_interesting_location = (
            (taxi_location[0] == passenger_location[0] and taxi_location[1] == passenger_location[1]) or (
            taxi_location[0] == passenger_destination[0] and taxi_location[1] == passenger_destination[1]))
    passenger_in_interesting_location = passenger_location in passenger_origins
    # valid_passenger_destination = ((passenger_destination[0] == passenger_location[0]) and (
    #         passenger_destination[1] == passenger_location[1]) and passenger_status > 2) or (
    #                                       passenger_destination[0] != passenger_location[0]) or (
    #                                       passenger_destination[1] != passenger_location[1])

    if fuel_is_full and taxi_in_interesting_location and passenger_in_interesting_location:  # and valid_passenger_destination:
        return True
    return False


def get_possible_passenger_origins(env):
    return env.passengers_locations


def get_possible_passenger_destinations(env):
    return env.passengers_locations


def get_automatic_anticipated_policy_from_agent(env, agent_for_policy_generator, num_of_episodes,
                                                num_states_in_partial_policy):
    """
    get automatic anticipated policy from given agent
    """

    # create agent
    agent = rl_agent.create_agent(env, agent_for_policy_generator)
    # train the agent in the environment
    train_episode_reward_mean = rl_agent.run(agent, num_of_episodes, method=TRAIN)

    policy_dict = agent.policy_dict

    automatic_anticipated_policy = sample_anticipated_policy(policy_dict, env, num_states_in_partial_policy)
    return automatic_anticipated_policy
