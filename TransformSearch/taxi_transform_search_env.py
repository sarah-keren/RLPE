from Environments.SingleTaxiEnv.single_taxi_wrapper import *
from utils import *
from save_load_utils import *

ACT = "action"
IDX = "idx"
VAL = "val"


class SingleTaxiTransformSearchEnv(SingleTaxiSimpleEnv):
    def __init__(self):
        super().__init__()
        self.action_translation_dict = {}
        new_act = self.num_actions - 1
        preconditions = load_pkl_file(PRECONDITIONS_PATH)
        all_single_transformed_envs = load_pkl_file(ALL_SMALL_TAXI_TRANSFORMED_ENVS_PATH)
        for act, act_info in preconditions.not_allowed_features.items():
            for pre_idx, pre_val in act_info.items():
                for val in pre_val:
                    new_act += 1
                    transform_name = f"{act}_{pre_idx}_{val}"
                    if transform_name not in all_single_transformed_envs.keys():
                        continue
                    transformed_env = all_single_transformed_envs[transform_name]
                    self.action_translation_dict[new_act] = transform_name
                    for s, s_info in self.P.items():
                        self.P[s][new_act] = transformed_env.P[s][act]
        self.num_actions = new_act + 1
        discrete.DiscreteEnv.__init__(self, self.num_states, self.num_actions, self.P, self.initial_state_distribution)


class SingleTaxiTransformAnticipatedSearchEnv(SingleTaxiSimpleEnv):
    def __init__(self, anticipated_policy):
        super().__init__()
        self.action_translation_dict = {}
        new_act = self.num_actions - 1
        basic_actions = [_ for _ in range(self.num_actions)]
        preconditions = load_pkl_file(PRECONDITIONS_PATH)
        all_single_transformed_envs = load_pkl_file(ALL_SMALL_TAXI_TRANSFORMED_ENVS_PATH)
        decoded_anticipated_states, anticipated_actions = anticipated_policy.keys(), anticipated_policy.values()
        # flatten_anticipated_actions = [item for sublist in anticipated_actions for item in sublist]
        for state, state_info in self.P.items():
            new_act = self.num_actions - 1
            align_with_anticipated, anticipated_act, _ = is_state_align_with_anticipated_policy(self, state,
                                                                                                anticipated_policy)
            for not_allowed_act, not_allowed_act_info in preconditions.not_allowed_features.items():
                for pre_idx, pre_val in not_allowed_act_info.items():
                    for val in pre_val:
                        transform_name = f"{not_allowed_act}_{pre_idx}_{val}"
                        if transform_name not in all_single_transformed_envs.keys():
                            continue
                        new_act += 1
                        transformed_env = all_single_transformed_envs[transform_name]
                        self.action_translation_dict[new_act] = {ACT: not_allowed_act, IDX: pre_idx, VAL: val}
                        if align_with_anticipated:
                            temp_info = list(transformed_env.P[state][not_allowed_act][0])
                            if new_act == 24 and state == 8950:  # TODO - its a patch! to delete!
                                temp_info[3] = True
                                self.P[state][new_act] = [tuple(temp_info)]
                            else:
                                self.P[state][new_act] = transformed_env.P[state][not_allowed_act]
                        else:
                            temp_info = list(transformed_env.P[state][not_allowed_act][0])
                            temp_info[1] = state
                            temp_info[2] = temp_info[2] if temp_info[2] < 0 else -1.0
                            self.P[state][new_act] = [tuple(temp_info)]

            if align_with_anticipated:
                next_anticipated_state = self.P[state][anticipated_act][0][1]
                if state != next_anticipated_state:
                    relevant_actions = basic_actions
                else:
                    relevant_actions = [k for (k, v) in self.action_translation_dict.items() if
                                        v[ACT] == anticipated_act]
                    relevant_actions.append(anticipated_act)
                for other_act, other_act_info in self.P[state].items():
                    if other_act not in relevant_actions:
                        temp_info = list(other_act_info[0])
                        temp_info[1] = state
                        temp_info[2] = temp_info[2] if temp_info[2] < 0 else -1.0
                        self.P[state][other_act] = [tuple(temp_info)]
        self.num_actions = new_act + 1
        discrete.DiscreteEnv.__init__(self, self.num_states, self.num_actions, self.P, self.initial_state_distribution)


def create_transform_search_taxi_env(env, env_preconditions, anticipated_policy):
    default_data = [1.0, None, -1, False]
    translation_action_to_precondition_dict = dict()
    new_act_num = env.num_actions - 1
    for action in range(env.num_actions):
        action_preconditions = env_preconditions[action]
        for pre in action_preconditions:
            new_act_num += 1
            for s in env.P.keys():
                default_data[NEXT_STATE_IDX] = s
                env.P[s][new_act_num] = [tuple(default_data)]
                translation_action_to_precondition_dict[pre[PRECONDITION_NAME_IDX]] = new_act_num
    env.num_actions = new_act_num + 1
    discrete.DiscreteEnv.__init__(env, env.num_states, env.num_actions, env.P, env.initial_state_distribution)
    for anticipated_s, anticipated_actions in anticipated_policy.items():
        anticipated_states = env.get_states_from_partial_obs(anticipated_s)
        for anticipated_a in anticipated_actions:
            for anticipated_state in anticipated_states:
                tmp_info = list(env.P[anticipated_state][anticipated_a][0])
                next_state = tmp_info[NEXT_STATE_IDX]
                if next_state != anticipated_state:
                    for act in range(env.num_actions):
                        if act not in anticipated_actions:
                            tmp_info = list(env.P[anticipated_state][act][0])
                            tmp_info[NEXT_STATE_IDX] = anticipated_state
                            env.P[anticipated_state][act] = [tuple(tmp_info)]
                else:
                    relevant_preconditions = env_preconditions[anticipated_a]
                    for precondition in relevant_preconditions:
                        _, transformed_env = load_transform_by_name(precondition[PRECONDITION_NAME_IDX])
                        new_pre_action = translation_action_to_precondition_dict[precondition[PRECONDITION_NAME_IDX]]
                        env.P[anticipated_state][new_pre_action] = transformed_env.P[anticipated_state][anticipated_a]
    env.translation_action_to_precondition_dict = translation_action_to_precondition_dict
    return env


def reformat_preconditions(env_preconditions):
    new_preconditions = dict()
    for action, action_info in env_preconditions.not_allowed_features.items():
        for idx, values in action_info.items():
            for val in values:
                precondition_name = f"{action}_{idx}_{val}"
                if action in new_preconditions.keys():
                    new_preconditions[action].append((precondition_name, idx, val))
                else:
                    new_preconditions[action] = [(precondition_name, idx, val)]
    return new_preconditions

    # if __name__ == '__main__':
#     anticipated_policy = ANTICIPATED_POLICY
#     search_env = SingleTaxiTransformAnticipatedSearchEnv(anticipated_policy)
# file_name = "taxi_example_data/taxi_transformed_env/search_env_without_5.pkl"
# save_pkl_file(file_name, search_env)
#
# transform_name = "search_anticipated_env"
# save_pkl_file(transform_name, search_env)
# file = open(file_name, "rb")
# new_env = pickle.load(file)
# generate_agent(SINGLE_TAXI_EXAMPLE, KERAS_DQN, 100, new_env, transform_name)
# actions = [1, 1, 19, 7, 9, 9, 7, 24]
# all_reward = 0
# for act in actions:
#     search_env.render()
#     next_s, r, d, prob = search_env.step(act)
#     all_reward += r
#     print(f"state:{search_env.decode(next_s)}")
#     print(f"reward:{r} done:{d} prob:{prob}")
#     print(f"all_reward:{all_reward}")
# a = 7
