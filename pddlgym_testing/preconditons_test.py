import pddlgym.pddlgym as pddlgym
import imageio
import matplotlib.pyplot as plt
import os
from pddl_tools.pddl_utils import *
from itertools import combinations

ENV_NAME = "PDDLEnvSokoban-v0"


def get_precondition_file_name(preconditions_relaxation):
    pre_name_arr = []
    for pre in preconditions_relaxation:
        tmp_arr = pre[1:-1].split(" ?")
        new_name = ''.join(tmp_arr)
        pre_name_arr.append(new_name)
    return '_'.join(pre_name_arr)


env = pddlgym.make(ENV_NAME)
env.reset()
# env.problem_dir
problem_directory = env._problem_dir
domain_file = env._domain_file

curr_dir = os.path.join(os.getcwd(), "relaxed_envs")
pddl_example = read_pddl(domain_file)  # reads or example file
all_preconditions = get_pddl_pred(pddl_example)

env = pddlgym.make(ENV_NAME)
possible_indices = []
for i in range(1, len(all_preconditions)):
    possible_idx = [list(_) for _ in combinations([_ for _ in range(len(all_preconditions))], i)]
    possible_indices += possible_idx

error_preconditions = dict()
for idx_comb in possible_indices:
    preconditions_relax = [pre for (i, pre) in enumerate(all_preconditions) if i in idx_comb]
    # print("to remove pre list", some_preconditions)

    new_pddl = relax_preconditions(pddl_example, preconditions_relax)
    if not new_pddl:
        continue
    recon_file_name = get_precondition_file_name(preconditions_relax)
    relax_file_path = os.path.join(curr_dir, f"{recon_file_name}.pddl")
    save_pddl_file(new_pddl, relax_file_path)

    # env.problem_dir
    env.__init__(relax_file_path, problem_directory)

    actions = set()
    obs, _ = env.reset()
    #  just for discovering actions space
    for i in range(100):
        actions.add(env.action_space.sample(obs))

    actions = list(actions)
    for act in actions:
        try:
            obs, reward, done, debug_info = env.step(act)
        except:
            # print(f"ACTION: {act} IDX: {idx_comb}")
            # print(f"{some_preconditions}")
            if act in error_preconditions:
                error_preconditions[act].append((idx_comb, preconditions_relax))
            else:
                error_preconditions[act] = [(idx_comb, preconditions_relax)]
            continue
a = 7
