import os
import shutil
import pickle

from Agents.RL_agents import rl_agent
from constants import *
import pddlgym.pddlgym as pddlgym
from pddl_parser.planner import Planner
from pddl_tools.pddl_utils import *
import time


def get_env(env_name, operators_as_actions=False):
    env = pddlgym.make("PDDLEnv{}-v0".format(env_name.capitalize()), operators_as_actions=operators_as_actions)
    env.fix_problem_index(ENV_IDX)
    obs = env.reset()
    env_action_space = env.action_space
    actions = set()
    for i in range(1):
        actions.add(env.sample_action_space(obs))
    num_states = 600000
    setattr(env_action_space, "n", len(env.action_space._all_ground_literals))
    setattr(env, "actions", env.action_space._all_ground_literals)
    setattr(env, "num_states", num_states)
    return env


def get_anticipated_policy(env, env_name):
    problem_idx = env._problem_idx + 1
    planner = Planner()
    plan = planner.solve(f'{PDDL_PROBLEM_PATH}{env_name}.pddl',
                         f'{PDDL_PROBLEM_PATH}{env_name}/problem0{problem_idx}.pddl', )
    return plan


def run_agent_on_env(env, agent_name, env_name):
    agent = rl_agent.create_agent(env, agent_name)
    print(f"\nTraining and evaluating the {agent_name} on \"{env_name}\" environment")
    train_result, evaluation_result = None, None
    try:
        train_result = rl_agent.run(agent, num_of_episodes=ITER_NUM, method=TRAIN)
        evaluation_result = rl_agent.run(agent, num_of_episodes=ITER_NUM, method=EVALUATE)
    except Exception as e:
        print("Failed to train the agent on \"{env_name}\" environment")
        print(e)
    return agent, train_result, evaluation_result


def get_agents_policy(env, agent):
    prev_state = env.reset()
    done = False
    policy = []
    i = 0
    while not done and i < 100:
        action = agent.compute_action(prev_state)
        after_state, reward, done, debug_info = env.step(action)
        # decoded_prev_state = env.get_state_from_idx(prev_state)
        decoded_action = env.actions[action]
        # decoded_after_state = env.get_state_from_idx(after_state)
        policy.append(decoded_action)
        i += 1
    return policy, done


def is_anticipated_policy_achieved(anticipated_policy, agent_policy, agent_done):
    if anticipated_policy and agent_done:
        satisfied_actions, unsatisfied_actions = 0, 0
        for anticipated_act, actor_act in zip(anticipated_policy, agent_policy):
            anticipated_direction = anticipated_act.parameters[5].split('-')[1]
            actor_direction = str(actor_act.variables[0]).split('-')[1].split(':')[0]
            # if anticipated_act.name == actor_act.predicate.name:
            if anticipated_direction == actor_direction:
                satisfied_actions += 1
            else:
                unsatisfied_actions += 1
        score = satisfied_actions / (satisfied_actions + unsatisfied_actions)
        return score == 1.0, score
    else:
        return False, 0.0


def get_relaxed_env(curr_dir, env_name, new_pddl, original_env, preconditions_relax, problem_directory):
    recon_file_name = get_precondition_file_name(preconditions_relax)
    relax_file_path = os.path.join(curr_dir, f"{recon_file_name}.pddl")
    save_pddl_file(new_pddl, relax_file_path)
    relaxed_env = get_env(env_name)
    relaxed_env.__init__(relax_file_path, problem_directory, render=original_env.render, operators_as_actions=False)
    relaxed_env.fix_problem_index(ENV_IDX)
    obs = relaxed_env.reset()
    env_action_space = relaxed_env.action_space
    actions = set()
    for i in range(1):
        actions.add(relaxed_env.sample_action_space(obs))
    num_states = 600000
    setattr(env_action_space, "n", len(relaxed_env.action_space._all_ground_literals))
    setattr(relaxed_env, "actions", relaxed_env.action_space._all_ground_literals)
    setattr(relaxed_env, "num_states", num_states)
    return relax_file_path, relaxed_env


def run_base_method(env_name, agent_name):
    times = []
    result = dict()
    start_time = time.time()

    original_env = get_env(env_name)

    # anticipated_policy = get_anticipated_policy(original_env, env_name)

    agent, train_result, evaluation_result = run_agent_on_env(original_env, agent_name, env_name)

    agent_policy, agent_done = get_agents_policy(original_env, agent)

    # policy_satisfaction, satisfaction_score = is_anticipated_policy_achieved(anticipated_policy, agent_policy,
    #                                                                          agent_done)

    t = time.time() - start_time
    times.append((ORIGINAL_ENV, t))
    result[ORIGINAL_ENV] = {"env": original_env, "agent": agent, "time": t}
    # if policy_satisfaction:
    #     print("The anticipated policy is satisfied, no explanation needed!")
    #     return original_env, agent, time.time() - start_time, ORIGINAL_ENV, result

    original_pddl, all_preconditions, possible_indices = get_precondition_relaxations(original_env)

    original_agent = agent
    problem_directory = original_env._problem_dir
    curr_dir = os.path.join(os.getcwd(), "relaxed_envs")
    for idx_comb in possible_indices:
        start_time = time.time()
        preconditions_relax = [pre for (i, pre) in enumerate(all_preconditions) if i in idx_comb]

        new_pddl = relax_preconditions(original_pddl, preconditions_relax)
        if not new_pddl:
            continue
        relax_file_path, relaxed_env = get_relaxed_env(curr_dir, env_name, new_pddl, original_env, preconditions_relax,
                                                       problem_directory)
        relax_name = relax_file_path.split('\\')[-1][:-5]
        agent, train_result, evaluation_result = run_agent_on_env(relaxed_env, agent_name, relax_name)

        agent_policy, agent_done = get_agents_policy(relaxed_env, agent)

        # policy_satisfaction, satisfaction_score = is_anticipated_policy_achieved(anticipated_policy, agent_policy,
        #                                                                          agent_done)
        # policy_satisfaction = False
        delete_pddl_file(relax_file_path)
        t = time.time() - start_time
        times.append((preconditions_relax, t))
        result[relax_name] = {"env": relaxed_env, "agent": agent, "time": t, "agent_done": agent_done}
        # if policy_satisfaction:
        #     print("We found transform!")
        #     print(preconditions_relax)
        #     return relaxed_env, agent, t, preconditions_relax, result
    return result


def run_rlpe(agent_name, env_name, search_method, num_of_epochs):
    if os.path.isdir(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)
    if search_method == BASE:
        result = run_base_method(env_name, agent_name)
        # save result
        pkl_name = f"{DATA_FOLDER}{agent_name}_{BASE}_all_stats_states.pkl"
        output = open(pkl_name, 'wb')
        pickle.dump(result, output)
        output.close()
    elif search_method == PRE_TRAIN:
        pass
    elif search_method == CLUSTER:
        pass
    else:
        pass

    return result
