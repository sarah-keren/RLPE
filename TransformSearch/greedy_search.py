import copy
from collections import defaultdict

from Transforms.single_taxi_transforms import generate_transformed_env


def cluster_transforms_by_features(preconditions):
    feature_clusters = defaultdict(dict)  # {action: {idx : [[values]]}}
    for action, action_info in preconditions.items():
        for idx, values in action_info.items():
            for val in values:
                if action in feature_clusters[idx].keys():
                    if idx in feature_clusters[idx][action].keys():
                        feature_clusters[idx][action][idx].append(list(val))
                    else:
                        feature_clusters[idx][action][idx] = [list(val)]
                else:
                    feature_clusters[idx][action] = {idx: [list(val)]}
    return feature_clusters


def calculate_possible_satisfaction_rate(transformed_env, anticipated_policy):
    pos, neg, = 0, 0
    for anticipated_state, anticipated_actions in anticipated_policy.items():
        states_from_anticipated_state = transformed_env.get_states_from_partial_obs(anticipated_state)
        for state in states_from_anticipated_state:
            transformed_env.s = state
            for anticipated_action in anticipated_actions:
                next_state, reward, done, info = transformed_env.step(anticipated_action)
                if next_state != state:
                    pos += 1
                else:
                    neg += 1
    all_res = pos + neg
    return pos / all_res


def refactor_max_cluster(max_cluster):
    max_cluster_list = []
    for act, act_info in max_cluster.items():
        for idx, values in act_info.items():
            for val in values:
                max_cluster_list.append((act, idx, val))
    return max_cluster_list


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def dict_intersection(working_cluster, temp_cluster):
    working_cluster_list = refactor_max_cluster(working_cluster)
    temp_cluster_list = refactor_max_cluster(temp_cluster)
    cluster_intersection = intersection(working_cluster_list, temp_cluster_list)
    new_working_cluster = defaultdict(dict)
    for action, idx, val in cluster_intersection:
        if action in new_working_cluster.keys():
            if idx in new_working_cluster[action].keys():
                new_working_cluster[action][idx].append(list(val))
            else:
                new_working_cluster[action][idx] = [list(val)]
        else:
            new_working_cluster[action] = {idx: [list(val)]}
    return new_working_cluster


def greedy_search(preconditions, anticipated_policy):
    clusters = cluster_transforms_by_features(preconditions.not_allowed_features)
    transformed_envs1, satisfaction_rates1, transformed_envs2, satisfaction_rates2 = [], [], [], []
    max_transformed_env_opt, best_cluster = None, None
    k, msr_opt = 0, 0.0
    while k < 100 and msr_opt != 1.0:
        msr, max_cluster, max_transformed_env = 0, None, None
        for cluster in clusters.values():
            transformed_env = generate_transformed_env(cluster, "", save=False, try_load_existing=False)
            satisfaction_rate = calculate_possible_satisfaction_rate(transformed_env, anticipated_policy)
            satisfaction_rates1.append(satisfaction_rate)
            transformed_envs1.append(transformed_env)
            if msr < satisfaction_rate:
                msr = satisfaction_rate
                max_cluster = cluster
                max_transformed_env = transformed_env
        working_cluster = copy.deepcopy(max_cluster)
        max_cluster_list = refactor_max_cluster(max_cluster)
        msr_opt, max_transformed_env_opt = 1.0, None
        for transform in max_cluster_list:
            temp_cluster = copy.deepcopy(max_cluster)
            temp_cluster[transform[0]][transform[1]].remove(transform[2])
            if len(temp_cluster[transform[0]][transform[1]]) == 0:
                del temp_cluster[transform[0]][transform[1]]
                if len(temp_cluster[transform[0]]) == 0:
                    del temp_cluster[transform[0]]
            transformed_env = generate_transformed_env(temp_cluster, "", save=False, try_load_existing=False)
            transformed_envs2.append(transformed_env)
            satisfaction_rate = calculate_possible_satisfaction_rate(transformed_env, anticipated_policy)
            satisfaction_rates2.append(satisfaction_rate)
            if msr_opt <= satisfaction_rate:
                msr_opt = satisfaction_rate
                working_cluster = dict_intersection(working_cluster, temp_cluster)
                max_transformed_env_opt = transformed_env
                best_cluster = working_cluster
        k += 1
    return max_transformed_env_opt, best_cluster
