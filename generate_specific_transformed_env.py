from Transforms.single_taxi_transforms import *
from utils import *


def generate_specific_transformed_env():
    #     env_default_values = [0, 0, 0, 1, MAX_FUEL - 1]
    # a_file = open("taxi_example_data/small_taxi_env_preconditions.pkl", "rb")
    # cur_env_preconditions = pickle.load(a_file)

    act1, act2, act3, act4, act5 = 0, 1, 2, 4, 5
    pre_idx = (4,)
    pre_val = [0]
    dir_name = TRANSFORMS_PATH
    env_file_name = f"{act1}_{pre_idx}_{pre_val}_{act2}_{pre_idx}_{pre_val}_{act3}_{pre_idx}_{pre_val}_{act4}_{pre_idx}_{pre_val}_{act5}_{pre_idx}_{pre_val}.pkl"
    precondition = {act1: {pre_idx: pre_val},
                    act2: {pre_idx: pre_val},
                    act3: {pre_idx: pre_val},
                    act4: {pre_idx: pre_val},
                    act5: {pre_idx: pre_val},
                    }
    env = generate_transformed_env(precondition, env_file_name=dir_name + env_file_name, save=True)

    print("DONE!")
    return env


def create_data_structure_with_all_envs():
    data_path = TRAINED_AGENT_ON_SPECIFIC_TRANSFORM_NUM_PATH
    all_single_transformed_envs = {}
    for file_name in os.listdir(data_path):
        transform_name = file_name[:-4]
        file = open(data_path + file_name, "rb")
        new_env = pickle.load(file)
        all_single_transformed_envs[transform_name] = new_env
    return all_single_transformed_envs


if __name__ == '__main__':
    #     all_single_transformed_envs = create_data_structure_with_all_envs()
    transformed_env = generate_specific_transformed_env()
    a = 7
