import copy
import itertools
import os
import pickle
import re
from Agents.RL_agents import rl_agent
from Transforms.transform_constants import SINGLE_TAXI_EXAMPLE
from constants import *
import warnings


def make_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except OSError:
        print("Directory %s exists already" % dir_name)


def save_pkl_file(file_name, data):
    if file_name[-4:] != ".pkl":
        file_name = file_name + ".pkl"
    file = open(file_name, 'wb')
    pickle.dump(data, file)
    file.close()


def load_pkl_file(file_path):
    if file_path[-4:] != ".pkl":
        file_path = file_path + ".pkl"
    try:
        a_file = open(file_path, "rb")
        obj = pickle.load(a_file)
    except:
        warnings.warn(f"ERROR: cannot load object from file path: {file_path}")
        obj = None
    return obj


def save_trained_model(agent, agent_name, transform_name):
    model_name = agent_name + '_' + transform_name
    dir_path = TRAINED_AGENT_SAVE_PATH
    make_dir(dir_path + model_name)
    dir_path = dir_path + model_name + '/'
    agent.model.save_weights(dir_path + model_name)


def make_or_restore_model(env, agent_name, transform_name):
    # Prepare a directory to store all the checkpoints.
    checkpoint_dir_path = TRAINED_AGENT_SAVE_PATH + f"{agent_name}_{transform_name}/"
    if not os.path.exists(checkpoint_dir_path):
        os.makedirs(checkpoint_dir_path)
    # Either restore the latest model, or create a fresh one if there is no checkpoint available.
    checkpoints = [checkpoint_dir_path + name for name in os.listdir(checkpoint_dir_path)]
    restored = True if checkpoints else False
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)

        return load_existing_agent(env, agent_name, transform_name), restored
    print("Creating a new model")
    return rl_agent.create_agent(env, agent_name), restored


def load_existing_results(agent_name, env_name, num_episodes=ITER_NUM, dir_path=TRAINED_AGENT_RESULTS_PATH):
    info_name = f"{env_name}_{agent_name}"
    file_path = dir_path + info_name + f"_{num_episodes}/"
    try:
        explanation = load_pkl_file(file_path + info_name + "_explanation.pkl")
        result = load_pkl_file(file_path + info_name + "_result.pkl")
    except:
        warnings.warn(f"ERROR: Not available results for env_name: {env_name}, agent_name: {agent_name}")
        result, explanation = [], []

    return result, explanation


def load_env_preconditions(env_name):
    if env_name == SINGLE_TAXI_EXAMPLE:
        a_file = open(PRECONDITIONS_PATH, "rb")
        preconditions = pickle.load(a_file)
        return preconditions
    else:
        raise Exception("not valid env_name")


def load_existing_transforms_from_dir(working_dir=TRANSFORMS_PATH):
    possible_env_files = os.listdir(working_dir)
    cur_transforms = dict()
    dict_idx = 0
    for i, file_name in enumerate(possible_env_files):
        transform_name, new_env = load_transform_by_name(file_name, dir_name=working_dir)
        cur_transforms[dict_idx] = transform_name, new_env
        dict_idx += 1
    global transforms
    transforms = cur_transforms
    return cur_transforms


def get_precondition_actions_from_string(pre_string):
    string_for_extracting_actions = copy.deepcopy(pre_string)
    string_for_extracting_actions = re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", string_for_extracting_actions)
    precondition_actions = [int(s) for s in string_for_extracting_actions.split('_') if s.isdigit()]
    return precondition_actions


def get_precondition_idx_from_string(single_pre_string):
    pre_idx = single_pre_string[single_pre_string.find("(") + 1:single_pre_string.find(")")]
    pre_idx = tuple(int(pre) for pre in pre_idx if pre != ' ' and pre != ',')
    return pre_idx


def get_precondition_val_from_string(single_pre_string):
    pre_val = single_pre_string[single_pre_string.find("[") + 1:single_pre_string.find("]")]
    pre_val = [int(pre) for pre in pre_val if pre != ' ']
    return pre_val


def load_existing_transforms_by_anticipated_policy(env_name, anticipated_policy, working_dir=TRANSFORMS_PATH):
    possible_env_files = os.listdir(working_dir)
    cur_transforms = dict()
    dict_idx = -1
    anticipate_policy_actions = anticipated_policy.values()
    all_possible_anticipated_policies = list(
        itertools.combinations(anticipate_policy_actions, len(anticipate_policy_actions)))
    for i, file_name in enumerate(possible_env_files):
        precondition_actions = get_precondition_actions_from_string(file_name)
        for policy_action_list in all_possible_anticipated_policies:
            policy_action_list = [p for tmp in policy_action_list for p in tmp]
            is_precondition_actions_relevant = any(
                pre_action in policy_action_list for pre_action in precondition_actions)
            if is_precondition_actions_relevant:
                dict_idx += 1
                # precondition_idx = file_name[file_name.find("(") + 1:file_name.find(")")]
                # precondition_val = file_name[file_name.find("[") + 1:file_name.find("]")]
                transform_name, new_env = load_transform_by_name(file_name)
                cur_transforms[dict_idx] = transform_name, new_env
    global transforms
    transforms = cur_transforms
    return cur_transforms


def load_transform_by_name(file_name, dir_name=TRANSFORMS_PATH):
    if not file_name[:-4] == ".pkl":
        file_name += ".pkl"
    transform_name = os.path.basename(file_name)[:-4]
    try:
        file = open(dir_name + file_name, "rb")
        new_env = pickle.load(file)
    except:
        warnings.warn(f"[save_load_utils][load_transform_by_name]: could not load existing transform")
        new_env = None
    return transform_name, new_env


def load_existing_agent(env, agent_name, transform_name, trained_agents_path=TRAINED_AGENT_SAVE_PATH):
    model_name = agent_name + '_' + transform_name
    dir_path = trained_agents_path + model_name + '/'
    new_agent = rl_agent.create_agent(env, agent_name)
    try:
        new_agent = new_agent.load_existing_agent(dir_path + model_name)
    except:
        warnings.warn(f"ERROR: bad agent: {dir_path + model_name}")
        new_agent = None
    return new_agent
