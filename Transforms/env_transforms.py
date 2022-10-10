import collections
import numpy as np
from save_load_utils import *


def preconditions_relaxation(preconditions_info, env):
    # preconditions = EnvPreconditions(env)
    a_file = open(PRECONDITIONS_PATH, "rb")
    preconditions = pickle.load(a_file)

    diff_dict, state_by_diff, next_state_by_action = get_diff_for_actions(env.P, env)
    # a_file = open("taxi_example_data/diff_dict.pkl", "rb")
    # diff_dict = pickle.load(a_file)

    for act, act_info in preconditions_info.items():
        if env.deterministic:  # deterministic case
            pre_process_info_and_update_p(env, act, diff_dict, act_info, act, preconditions, None)
        else:
            for act_prob, pre_info in act_info.items():
                pre_process_info_and_update_p(env, act_prob, diff_dict, pre_info, act, preconditions, act_prob)
    return preconditions


def pre_process_info_and_update_p(env, act_to_diff, diff_dict, pre_info, act, preconditions, act_prob=None):
    state_to_replace = get_state_to_replace(env, pre_info, act, preconditions)
    mapping_states_dict = get_mapping_states_dict(env, state_to_replace, diff_dict[act_to_diff])
    update_p_matrix_by_relax_preconditions(env, mapping_states_dict, act, act_prob)


def get_state_to_replace(env, preconditions_info, act, preconditions):
    state_to_replace = []
    for state in env.P.keys():
        decoded_state = np.array(env.decode(state))
        for idx, values in preconditions_info.items():
            for v in values:
                state_values = decoded_state[np.array([idx])]
                sufficient_for_action = is_sufficient_for_action(idx, v, preconditions.not_allowed_features, act,
                                                                 decoded_state)
                if np.array([state_values == np.array(v)]).all() and sufficient_for_action:
                    state_to_replace.append(state)
    return state_to_replace


def is_sufficient_for_action(relaxed_idx, relaxed_val, not_allowed_features, act, decoded_state):
    result = True
    for key, values in not_allowed_features[act].items():
        for val in values:
            if np.array([decoded_state[list(key)] == val]).all() and (
                    np.array([relaxed_idx != key]).all() or np.array([relaxed_val != val]).all()):
                result = False
                break
        if not result:
            break
    return result


def is_state_need_a_replacement(act, preconditions, preconditions_info, act_prob):
    if act_prob:
        allowed_features_by_action = preconditions.allowed_features[act][act_prob]
    else:
        allowed_features_by_action = preconditions.allowed_features[act]


def get_mapping_states_dict(env, state_to_replace, optional_diff_vec):
    mapping_states_dict = dict((s, 0) for s in state_to_replace)
    for state in state_to_replace:
        decoded_state = np.array(env.decode(state))
        diff_vec = np.array([0] * len(decoded_state))
        if len(optional_diff_vec) > 3:
            for key, val in optional_diff_vec.items():
                val = list(val)[0]
                tmp_diff_vec = np.array(key)
                if (decoded_state + tmp_diff_vec[0])[val[0]] == val[1]:
                    diff_vec = tmp_diff_vec[0]
                    reward = tmp_diff_vec[1]
                    done = tmp_diff_vec[2]
                    break
                else:
                    reward = tmp_diff_vec[1]
                    done = tmp_diff_vec[2]
        else:
            diff_vec = np.array(optional_diff_vec[0][0])
            reward = optional_diff_vec[1]
            done = optional_diff_vec[2]
        not_legal_next_state = decoded_state + diff_vec
        state_is_legal, not_valid_idx = env.check_if_state_is_legal(not_legal_next_state, return_idxes=True)
        if not state_is_legal:
            new_not_legal_next_state = copy.deepcopy(not_legal_next_state)
            new_not_legal_next_state[not_valid_idx] = decoded_state[not_valid_idx]
            if (new_not_legal_next_state - decoded_state).any():
                mapping_states_dict[state] = (env.encode(*new_not_legal_next_state), reward, done)
            else:
                mapping_states_dict[state] = (state, reward, done)
        else:
            mapping_states_dict[state] = (env.encode(*not_legal_next_state), reward, done)
    return mapping_states_dict


def update_p_matrix_by_relax_preconditions(env, mapping_states_dict, pre_action, pre_prob_action=None):
    for state in mapping_states_dict:
        if pre_prob_action is not None:
            cur_info = list(env.P[state][pre_action][pre_prob_action])
            cur_info[1] = mapping_states_dict[state][0]
            cur_info[2] = mapping_states_dict[state][1]
            cur_info[3] = mapping_states_dict[state][2]
            env.P[state][pre_action][pre_prob_action] = cur_info
        else:  # deterministic case
            cur_info = list(env.P[state][pre_action][0])
            cur_info[1] = mapping_states_dict[state][0]
            cur_info[2] = mapping_states_dict[state][1]
            cur_info[3] = mapping_states_dict[state][2]
            env.P[state][pre_action] = [tuple(cur_info)]


def get_dicts(env, state, act_probs, act, next_state_dict_by_action, diff_dict_by_action, state_by_diff):
    next_state = act_probs[1]  # extract next state from the state prob
    reward = act_probs[2]
    done = act_probs[3]
    state_diff = np.array(env.decode(next_state)) - np.array(env.decode(state))
    if state_diff.any():
        next_state_dict_by_action[act].add((state, next_state))
        diff_dict_by_action[act].append([state_diff, reward, done])
        state_by_diff[act].add(state)
    return next_state_dict_by_action, diff_dict_by_action, state_by_diff


def get_diff_for_actions(p, env):
    next_state_by_action = [set() for act in range(len(p[0]))]
    diff_by_action = [list() for act in range(len(p[0]))]
    state_by_diff = [set() for act in range(len(p[0]))]
    for state, state_info in p.items():
        for act, act_probs in state_info.items():
            if len(act_probs) == 1:  # deterministic case
                next_state_by_action, diff_by_action, state_by_diff = get_dicts(env, state, act_probs[0], act,
                                                                                next_state_by_action, diff_by_action,
                                                                                state_by_diff)
            else:
                for act_prob, act_prob_info in enumerate(act_probs):
                    if len(act_prob_info) > 0:
                        next_state_by_action, diff_by_action, state_by_diff = get_dicts(env, state,
                                                                                        act_prob_info, act_prob,
                                                                                        next_state_by_action,
                                                                                        diff_by_action, state_by_diff)
    diff_dict = {}
    for i, action_diff in enumerate(diff_by_action):
        tmp_diff = [tuple(diff[0]) for diff in action_diff]
        occurrences = collections.Counter(tuple(tmp_diff))
        occurrences = occurrences.keys()
        reward = action_diff[0][1]
        done = action_diff[0][2] if (i != 5 and reward != 100) else True  # TODO - TEMPORARY! to change
        if len(occurrences) > 1:
            tmp_occ = list(occurrences)
            diff_dict[i] = dict(((x, reward, done), set()) for x in tmp_occ)
            for s, next_state in next_state_by_action[i]:
                for diff in tmp_occ:
                    decoded_state, decoded_next_state = np.array(env.decode(s)), np.array(env.decode(next_state))
                    state_diff = decoded_next_state - decoded_state
                    if not (state_diff - diff).any():
                        idx = np.nonzero(diff)
                        val = decoded_next_state[idx]
                        diff_dict[i][(diff, reward, done)].add((tuple(idx[0]), tuple(val)))
        else:
            diff_dict[i] = (list(occurrences), reward, done)
    return diff_dict, state_by_diff, next_state_by_action


def get_most_likely_outcome(p):
    for (s, s_probs) in p.items():
        for (a, a_probs) in s_probs.items():
            probs_list = [prob[0] for prob in a_probs]
            max_prob = max(probs_list)
            max_prob_idx = probs_list.index(max_prob)
            p[s][a] = [tuple([1.0] + list(p[s][a][max_prob_idx])[1:])]
    return p



