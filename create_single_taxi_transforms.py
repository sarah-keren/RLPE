import multiprocessing
from Agents.RL_agents.q_learning_agents import KERAS_DQN
from Transforms.single_taxi_transforms import *
from constants import *


def check_success_rate():
    success_rates = {}
    for file in os.listdir(TRAINED_AGENT_ON_SPECIFIC_TRANSFORM_NUM_PATH):
        success_rate = visualize_trained_agent(file)
        success_rates[file[:-4]] = success_rate

    for transform_name, success_rate in success_rates.items():
        if success_rate > 0.6:
            a = 7


def visualize_trained_agent(file_name):
    transform_name = file_name[:-4] if file_name[-4:] == ".pkl" else file_name
    file = open(TRAINED_AGENT_ON_SPECIFIC_TRANSFORM_NUM_PATH + file_name, "rb")
    new_env = pickle.load(file)

    agent_name = KERAS_DQN
    model_name = agent_name + '_' + transform_name
    dir_path = TRAINED_AGENTS_DIR_PATH + model_name + '/'
    new_agent = rl_agent.create_agent(new_env, agent_name)  # TODO - to fix this two lines
    new_agent = load_existing_agent(new_env, agent_name, transform_name, dir_path + model_name)
    evaluation_result = rl_agent.run(new_agent, 5, method=EVALUATE)

    result_name = transform_name + "_" + agent_name
    result_dir = result_name + "_200000/"
    file = open(TRAINED_AGENT_RESULTS_PATH + result_dir + result_name + "_explanation.pkl", "rb")
    explanation = pickle.load(file)
    file = open(TRAINED_AGENT_RESULTS_PATH + result_dir + result_name + "_result.pkl", "rb")
    result = pickle.load(file)

    return result[transform_name][SUCCESS_RATE]


def create_single_taxi_transforms():
    cur_env_preconditions = load_pkl_file(PRECONDITIONS_PATH)
    processes = []
    functions = [generate_single_transforms, generate_double_of_transforms, generate_triple_of_transforms]
    for func in functions:
        p = multiprocessing.Process(target=func, args=(cur_env_preconditions,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()