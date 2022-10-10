from Environments.SingleTaxiEnv.single_taxi_wrapper import *
from Transforms.env_transforms import *
from utils import *

DETERMINISTIC = True


class SingleTaxiTransformedEnv(SingleTaxiSimpleEnv):
    def __init__(self, transforms):
        super().__init__(DETERMINISTIC)
        self.state_visibility_transforms = transforms[STATE_VISIBILITY_TRANSFORM]
        self.all_outcome_determinization = transforms[ALL_OUTCOME_DETERMINIZATION]
        self.most_likely_outcome = transforms[MOST_LIKELY_OUTCOME]
        self.relax_preconditions = transforms[PRECONDITION_RELAXATION]

        if self.all_outcome_determinization:
            get_all_outcome_determinization_matrix(self.P)
        if self.most_likely_outcome:
            self.P = get_most_likely_outcome(self.P)
        if self.relax_preconditions:
            self.preconditions = preconditions_relaxation(self.relax_preconditions, self)
            self.preconditions_num = sum([len(self.preconditions.not_allowed_features[k]) for k in
                                          self.preconditions.not_allowed_features.keys()])

    def step(self, a):
        s, r, d, p = super(SingleTaxiTransformedEnv, self).step(a)
        next_state = self.decode(s)
        env_default_values = self.state_visibility_transforms[1]
        for i in self.state_visibility_transforms[0]:
            next_state[i] = env_default_values[i]

        transformed_next_state = self.encode(*next_state)
        return int(transformed_next_state), r, d, p


def get_all_outcome_determinization_matrix(p):
    # this is the function that converts the matrix!
    mapped_actions = {}
    ABNORMAL_STATES = [15]
    # states like the goal state, where the results are different, should be ignored in the action mapping
    # ABNORMAL_STATES should also include hole states but for now I have not included them because in this
    # particular case it doesn't matter

    for (s, s_outcomes) in p.items():
        new_outcomes = {}
        i = 0
        for (a, a_outcomes) in s_outcomes.items():
            for probs in a_outcomes:
                temp = (1.0,) + probs[1:]
                new_outcomes[i] = temp
                if s not in ABNORMAL_STATES:
                    mapped_actions[i] = a
                i += 1
        p[s] = new_outcomes


def get_single_taxi_transform_name(transforms):
    taxi_x_transform, taxi_y_transform = transforms[0], transforms[1]
    pass_loc_transform, pass_dest_transform = transforms[2], transforms[3]
    fuel_transform = transforms[4]
    all_outcome_determinization = transforms[5]
    most_likely_outcome = transforms[6]

    name = ""
    name += TAXI_LOC_X if taxi_x_transform else ""
    name += TAXI_LOC_Y if taxi_y_transform else ""
    name += PASS_LOC if pass_loc_transform else ""
    name += PASS_DEST if pass_dest_transform else ""
    name += FUEL if fuel_transform else ""
    name += ALL_OUTCOME_DETERMINIZATION if all_outcome_determinization else ""
    name += MOST_LIKELY_OUTCOME if most_likely_outcome else ""
    return name


def generate_triple_of_transforms(env_pre):
    dir_name = "triple_transform_envs"
    make_dir(TRAINED_AGENT_SAVE_PATH + dir_name)

    same_precondition = False
    for act1, pre1 in env_pre.not_allowed_features.items():
        for act2, pre2 in env_pre.not_allowed_features.items():
            for act3, pre3 in env_pre.not_allowed_features.items():
                for pre_idx1 in pre1.keys():
                    for pre_idx2 in pre2.keys():
                        for pre_idx3 in pre3.keys():
                            for pre_val1 in pre1[pre_idx1]:
                                for pre_val2 in pre2[pre_idx2]:
                                    for pre_val3 in pre3[pre_idx3]:
                                        if (act1 == act2 and (np.array(pre_idx1 == pre_idx2)).all() and np.array(
                                                (pre_val1 == pre_val2)).all()) or (act1 > act2) or ((act2 == act3 and (
                                                np.array(pre_idx2 == pre_idx3)).all() and np.array(
                                            (pre_val2 == pre_val3)).all()) or (act2 > act3)) or ((act1 == act3 and (
                                                np.array(pre_idx1 == pre_idx3)).all() and np.array(
                                            (pre_val1 == pre_val3)).all()) or (act1 > act3)):
                                            same_precondition = True
                                        if not same_precondition:

                                            env_file_name = f"{TRAINED_AGENT_SAVE_PATH}{dir_name}/{act1}_{pre_idx1}_{pre_val1}_{act2}_{pre_idx2}_{pre_val2}_{act3}_{pre_idx3}_{pre_val3}"
                                            if not os.path.exists(env_file_name + ".pkl"):
                                                print(f"act1: {act1} , pre_idx1: {pre_idx1} , pre_val1: {pre_val1}")
                                                print(f"act2: {act2} , pre_idx2: {pre_idx2} , pre_val2: {pre_val2}")
                                                print(f"act3: {act3} , pre_idx3: {pre_idx3} , pre_val3: {pre_val3}")
                                                print("\n")
                                                precondition = {act1: {pre_idx1: pre_val1},
                                                                act2: {pre_idx2: pre_val2},
                                                                act3: {pre_idx3: pre_val3}}
                                                generate_transformed_env(precondition, env_file_name, save=True)
                                            else:
                                                print(f"file: {env_file_name} already exists")
                                        same_precondition = False


def generate_double_of_transforms(env_pre):
    dir_name = "double_transform_envs"
    make_dir(TRAINED_AGENT_SAVE_PATH + dir_name)

    same_precondition = False
    for act1, pre1 in env_pre.not_allowed_features.items():
        for act2, pre2 in env_pre.not_allowed_features.items():
            for pre_idx1 in pre1.keys():
                for pre_idx2 in pre2.keys():
                    for pre_val1 in pre1[pre_idx1]:
                        for pre_val2 in pre2[pre_idx2]:
                            if (act1 == act2 and (np.array(pre_idx1 == pre_idx2)).all() and np.array(
                                    (pre_val1 == pre_val2)).all()) or (act1 > act2):
                                same_precondition = True
                            if not same_precondition:
                                print(f"act1: {act1} , pre_idx1: {pre_idx1} , pre_val1: {pre_val1}")
                                print(f"act2: {act2} , pre_idx2: {pre_idx2} , pre_val2: {pre_val2}")

                                env_file_name = f"{TRAINED_AGENT_SAVE_PATH}{dir_name}/{act1}_{pre_idx1}_{pre_val1}_{act2}_{pre_idx2}_{pre_val2}"
                                precondition = {act1: {pre_idx1: pre_val1},
                                                act2: {pre_idx2: pre_val2}}
                                generate_transformed_env(precondition, env_file_name, save=True)
                            same_precondition = False


def generate_single_transforms(env_preconditions):
    dir_name = "single_transform_envs"
    make_dir(TRAINED_AGENT_SAVE_PATH + dir_name)

    for act, preconditions in env_preconditions.not_allowed_features.items():
        for pre_idx in preconditions.keys():
            for pre_val in preconditions[pre_idx]:
                print(
                    f"calculating for act1: {act} , pre_idx1: {pre_idx} , pre_val1: {pre_val}")
                env_file_name = f"{TRAINED_AGENT_SAVE_PATH}{dir_name}/{act}_{pre_idx}_{pre_val}"
                precondition = {act: {pre_idx: pre_val}}
                generate_transformed_env(precondition, env_file_name, save=True)


def generate_transforms_to_depth(preconditions, anticipated_policy, max_depth=None):
    if not max_depth:
        max_depth = len(anticipated_policy)
    basic_preconditions_list = []
    for act, action_info in preconditions.items():
        for idx, values in action_info.items():
            for val in values:
                pre_string = f"{act}_{idx}_{val}"
                precondition_actions = get_precondition_actions_from_string(pre_string)
                influence = is_transform_actions_influence_anticipated_policy(list(anticipated_policy.values()),
                                                                              precondition_actions)
                if influence:
                    basic_preconditions_list.append(pre_string)
                    precondition = {act: {idx: val}}
                    env_file_name_path = f"{TRANSFORMS_PATH}" + pre_string
                    generate_transformed_env(precondition, env_file_name_path, try_load_existing=False, save=True)
    # basic_preconditions_list = load_pkl_file("basic_preconditions_list.pkl")

    all_preconditions_list = {1: tuple(tuple([pre]) for pre in basic_preconditions_list)}
    for i in range(2, max_depth):
        print(f"depth: {i}")
        all_preconditions_list[i] = tuple(itertools.combinations(basic_preconditions_list, i))
        for pre in all_preconditions_list[i]:
            precondition = dict()
            env_file_name = ""
            for p in pre:
                act = get_precondition_actions_from_string(p)[0]
                idx = get_precondition_idx_from_string(p)
                val = get_precondition_val_from_string(p)
                env_file_name += f"{act}_{idx}_{val}_"
                if act in precondition.keys():
                    if idx in precondition[act].keys():
                        precondition[act][idx].append(val)
                    else:
                        precondition[act][idx] = [val]
                else:
                    precondition[act] = {idx: [val]}
            env_file_name = env_file_name[:-1]
            env_file_name_path = f"{TRANSFORMS_PATH}" + env_file_name
            generate_transformed_env(precondition, env_file_name_path, save=True)


def generate_transformed_env(precondition, env_file_name, save=True, try_load_existing=True, env_default_values=None):
    print(f"Generating precondition env: {precondition}")
    new_env = None
    if try_load_existing:
        file_name = os.path.basename(env_file_name)
        transform_name, new_env = load_transform_by_name(file_name)
    if new_env is None:
        cur_transforms = {STATE_VISIBILITY_TRANSFORM: ([], env_default_values),
                          ALL_OUTCOME_DETERMINIZATION: False,
                          MOST_LIKELY_OUTCOME: False,
                          PRECONDITION_RELAXATION: precondition}
        new_env = SingleTaxiTransformedEnv(cur_transforms)
        if save:
            save_pkl_file(env_file_name + ".pkl", new_env)
    return new_env

# if __name__ == '__main__':
#     #     env_default_values = [0, 0, 0, 1, MAX_FUEL - 1]
#     a_file = open("taxi_example_data/small_taxi_env_preconditions.pkl", "rb")
#     cur_env_preconditions = pickle.load(a_file)
#
#     act1, act2, act3, act4, act5 = 0, 1, 2, 4, 5
#     pre_idx = (4,)
#     pre_val = [0]
#     dir_name = "taxi_example_data/taxi_transformed_env/"
#     env_file_name = f"{act1}_{pre_idx}_{pre_val}_{act2}_{pre_idx}_{pre_val}_{act3}_{pre_idx}_{pre_val}_{act4}_{pre_idx}_{pre_val}_{act5}_{pre_idx}_{pre_val}.pkl"
#     precondition = {act1: {pre_idx: pre_val},
#                     act2: {pre_idx: pre_val},
#                     act3: {pre_idx: pre_val},
#                     act4: {pre_idx: pre_val},
#                     act5: {pre_idx: pre_val},
#                     }
#     env = generate_transformed_env(precondition, env_file_name=dir_name + env_file_name, save=True)
#
#     print("DONE!")
