import sys
import numpy as np

from Agents.RL_agents.cem_agents import *
from Agents.RL_agents.rllib_agents import *
from Agents.RL_agents.q_learning_agents import *
from Agents.RL_agents.sarsa_agents import *

from Agents.value_iteration_agent import ValueIterationAgent

FORMAT_STRING = "\r{:3d} mean reward: {:6.2f}, variance: {:6.2f}, number of steps: {}"
from constants import *


def print_info(it, episode_result):
    print(FORMAT_STRING.format(it + 1, episode_result[EPISODE_REWARD_MEAN], episode_result[EPISODE_VARIANCE],
                               episode_result[EPISODE_STEP_NUM_MEAN]))


def get_rl_agent(agent_name, env, callbacks=None,  env_name=None, env_to_agent=None):  # TODO - fix the environment
    if is_rllib_agent(agent_name):
        agent = get_rllib_agent(agent_name, env_name, env, env_to_agent)
    elif agent_name == HANDS_ON_DQN:
        agent = HandsOnDQNAgent(env=env)
    elif agent_name == KERAS_DQN:
        agent = DQNKeras(env=env, callbacks=callbacks)
    elif agent_name == Q_LEARNING:
        agent = QLearningAgent(env=env)
    elif agent_name == KERAS_SARSA:
        agent = KerasSarsaAgent(env=env)
    elif agent_name == KERAS_CEM:
        agent = KerasCEMAgent(env=env)
    elif agent_name == VALUE_ITERATION:
        agent = ValueIterationAgent(env=env)
    else:
        raise Exception("Not valid agent name")
    return agent


def run(agent, num_of_episodes, method=TRAIN, print_process=True, visualize=True):
    result = {EPISODE_REWARD_MEAN: np.zeros(num_of_episodes), EPISODE_REWARD_MAX: np.zeros(num_of_episodes),
              EPISODE_REWARD_MIN: np.zeros(num_of_episodes), EPISODE_STEP_NUM_MEAN: np.zeros(num_of_episodes),
              EPISODE_VARIANCE: np.zeros(num_of_episodes)}
    print_rate = 100
    if print_process:
        message = "Training:" if method == TRAIN else "\nEvaluating:"
        print(message)
    if not isinstance(agent, DQNKeras): # and not isinstance(agent, KerasSarsaAgent): TODO - to add this!
        for it in range(num_of_episodes):
            episode_result = agent.run()
            if print_process and (it + 1) % print_rate == 0:
                print("\rEpisode {}/{}.".format(it + 1, num_of_episodes), end="")
                sys.stdout.flush()

            result[EPISODE_REWARD_MEAN][it] = sum(episode_result[EPISODE_REWARD_MEAN]) / len(episode_result[EPISODE_REWARD_MEAN])
            result[EPISODE_REWARD_MAX][it] = (episode_result[EPISODE_REWARD_MAX])
            result[EPISODE_REWARD_MIN][it] = (episode_result[EPISODE_REWARD_MIN])
            result[EPISODE_STEP_NUM_MEAN][it] = sum(episode_result[EPISODE_STEP_NUM_MEAN]) / len(episode_result[EPISODE_STEP_NUM_MEAN])
            result[EPISODE_VARIANCE][it] = (episode_result[EPISODE_VARIANCE])
        if method == EVALUATE:
            agent.evaluate(visualize)
    else:
        # result = agent.run()
        if method == EVALUATE:
            agent.evaluate(visualize)
        else:
            result = agent.run()
    return result


def run_episode(env, agent, method=TRAIN):
    state = env.reset()

    # == Initialize variables == #
    result = {EPISODE_REWARD_MEAN: 0.0, EPISODE_REWARD_MIN: np.inf, EPISODE_REWARD_MAX: -np.inf,
              EPISODE_STEP_NUM_MEAN: 0, EPISODE_VARIANCE: 0.0, TOTAL_EPISODE_REWARD: 0.0}
    total_reward = 0.0
    episode_len = 0
    for timestep in range(agent.timesteps_per_episode):
        if method == EVALUATE:
            env.render()
            pass

        action = agent.compute_action(state)  # Run Action

        next_state, reward, done, info = agent.env.step(action)  # Take action
        if method == EVALUATE:
            print(f"reward: {reward}, done: {done}, info: {info}")
        episode_len += 1

        total_reward += reward
        result[EPISODE_REWARD_MAX] = reward if reward > result[EPISODE_REWARD_MAX] else result[EPISODE_REWARD_MAX]
        result[EPISODE_REWARD_MIN] = reward if reward < result[EPISODE_REWARD_MIN] else result[EPISODE_REWARD_MIN]

        terminated = done
        if terminated:
            agent.stop_episode()
            break

        state = agent.episode_callback(state, action, reward, next_state, terminated)

    result[TOTAL_EPISODE_REWARD] = total_reward
    result[EPISODE_REWARD_MEAN] = total_reward / agent.timesteps_per_episode
    result[EPISODE_STEP_NUM_MEAN] = episode_len
    result[EPISODE_VARIANCE] = result[EPISODE_REWARD_MAX] - result[EPISODE_REWARD_MIN]
    return result


def create_agent(env, agent_name, callbacks=None, env_name=None):
    agent = get_rl_agent(agent_name, env, callbacks)
    return agent


# get the action performed by the agents in each observation
def get_policy_action(agent_rep, obs, reshape=False):
    # [taxi location], [current_fuel], [passengers_start_locations], [destinations], [passengers_status]
    if reshape:
        obs = np.reshape(obs, (1, len(obs)))
    action = agent_rep.compute_action(obs)
    return action


def get_policy_action_partial_obs(agent_rep, partial_obs, reshape=False):
    # [taxi location], [current_fuel], [passengers_start_locations], [destinations], [passengers_status]
    if reshape:
        partial_obs = np.reshape(partial_obs, (1, len(partial_obs)))
    action = agent_rep.compute_action(partial_obs)
    return action
