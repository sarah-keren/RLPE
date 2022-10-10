import multiprocessing
import shutil
from save_load_utils import *
from utils import *
from visualize import *
from Agents.RL_agents.q_learning_agents import *
import Agents.RL_agents.rl_agent as rl_agent
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.keras import backend as K
from TransformSearch.anticipation_BFS import *
from Transforms.env_precinditions import *


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])


def log(string):
    print(string)
    f = open('output_log.log', 'wb')
    f.write(string)
    f.close()


def create_run_and_evaluate_agent(original_env, transformed_env, agent_name, env_name, num_of_episodes,
                                  anticipated_policy, result, explanation):
    # GPUtil.showUtilization()
    agent, restored = make_or_restore_model(transformed_env, agent_name, env_name)

    print(f"\nTraining and evaluating the {agent_name} on \"{env_name}\" environment")
    if not restored:
        train_result = rl_agent.run(agent, num_of_episodes, method=TRAIN)
    else:
        train_result = None

    evaluation_result = rl_agent.run(agent, num_of_episodes, method=EVALUATE)

    anticipated_policy_achieved, success_rate = is_anticipated_policy_achieved(original_env, agent, anticipated_policy)

    result[env_name] = {EVALUATION_RESULTS: evaluation_result,
                        TRAINING_RESULTS: train_result,
                        SUCCESS_RATE: success_rate,
                        GOT_AN_EXPLANATION: False}

    if anticipated_policy_achieved:
        print(f"Environment {env_name} achieved an explanation!")
        explanation.append(env_name)
        result[env_name][GOT_AN_EXPLANATION] = True

    save_trained_model(agent, agent_name, env_name)
    return result, explanation


def run_multiprocess(original_env, transformed_env, agent_name, transform_name, num_of_episodes, anticipated_policy,
                     explanation):
    process_list = []
    NUM_GPUS = get_available_gpus()
    manager = multiprocessing.Manager()
    result = manager.dict()  # create shared data structure
    num_process_running = 0
    if num_process_running < NUM_GPUS:
        num_process_running += 1
        p = multiprocessing.Process(target=create_run_and_evaluate_agent, args=(
            original_env, transformed_env, agent_name, transform_name, num_of_episodes,
            anticipated_policy, result, explanation))
        process_list.append(p)
        p.start()
    else:
        for process in process_list:
            process.join()

        process_list = []
        num_process_running = 1
        p = multiprocessing.Process(target=create_run_and_evaluate_agent, args=(
            original_env, transformed_env, agent_name, transform_name, num_of_episodes, anticipated_policy, result,
            explanation))
        process_list.append(p)
        p.start()

    for name, res in result.items():
        if res["anticipated_policy_achieved"]:
            explanation.append(name)
            real_name = name
            result[real_name][GOT_AN_EXPLANATION] = True
    return result, explanation


def run_experiment(env_name, agent_name, num_of_episodes, num_states_in_partial_policy):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    K.set_session(session)

    result = {}
    original_env = get_env(env_name)

    # the anticipated policy (which is part of our input and defined by the user)
    # full_expert_policy_dict = get_expert(env_name, original_env).full_expert_policy_dict()
    # anticipated_policy = sample_anticipated_policy(full_expert_policy_dict, num_states_in_partial_policy)

    anticipated_policy = dict()
    anticipated_policy[(2, 0, 0, 3, None)] = [1]
    anticipated_policy[(1, 0, 0, 3, None)] = [1]
    anticipated_policy[(0, 0, 0, 3, None)] = [4]

    explanation = []

    # result, explanation = create_run_and_evaluate_agent(original_env, original_env, agent_name, ORIGINAL_ENV,
    #                                                     num_of_episodes,
    #                                                     anticipated_policy, result, explanation)
    transforms = dict()
    file_names = ['1_(4,)_[0]_4_(1,)_[2]_4_(0,)_[2].pkl', '4_(0,)_[2].pkl']
    for i, file_name in enumerate(file_names):
        transforms[i] = load_transform_by_name(file_name)
    # transforms = load_existing_transforms(env_name, anticipated_policy)

    for params, (transform_name, transformed_env) in transforms.items():
        result, explanation = run_multiprocess(original_env, transformed_env, agent_name, transform_name,
                                               num_of_episodes, anticipated_policy, explanation)
        # result, explanation = create_run_and_evaluate_agent(original_env, transformed_env, agent_name, transform_name,
        #                                                     num_of_episodes,
        #                                                     anticipated_policy, result, explanation)

    if explanation is None:
        print("no explanation found - you are too dumb for our system")
    else:
        print("explanation found %s:" % explanation)

    return result


def different_anticipated_policy_size_experiment(agent_name, env_name, num_of_epochs, num_of_episodes_per_epoch):
    # num_states = [1, 5, 10, 20, 30]
    num_states = [5]
    for num_states_in_partial_policy in num_states:
        default_experiment(agent_name, env_name, num_of_epochs, num_of_episodes_per_epoch, num_states_in_partial_policy)


def plot_results_by_number_of_transforms(pkl_name, output_folder, num_episodes):
    with open(pkl_name, 'rb') as f:
        data = pickle.load(f)
        result_1, result_2, result_3, result_4, result_5 = {}, {}, {}, {}, {}
        ls1, ls2, ls3, ls4, ls5 = [], [], [], [], []
        for cur_dict in data:
            for k, v in cur_dict.items():
                name = k.split('_')[:-1]
                if k == ORIGINAL_ENV:
                    orig_res = prepare_calc_mean(v)
                    result_1[k] = orig_res
                    result_2[k] = orig_res
                    result_3[k] = orig_res
                    result_4[k] = orig_res
                    result_5[k] = orig_res
                elif len(name) == 1:
                    result_1[k] = prepare_calc_mean(v)
                elif len(name) == 2:
                    result_2[k] = prepare_calc_mean(v)
                elif len(name) == 3:
                    result_3[k] = prepare_calc_mean(v)
                elif len(name) == 4:
                    result_4[k] = prepare_calc_mean(v)
                elif len(name) == 5:
                    result_5[k] = prepare_calc_mean(v)
            ls1.append(result_1)
            ls2.append(result_2)
            ls3.append(result_3)
            ls4.append(result_4)
            ls5.append(result_5)
            result_1, result_2, result_3, result_4, result_5 = {}, {}, {}, {}, {}
        save_cur_fig = False
        plot_results(ls1, output_folder, file_name="_1trans_" + str(num_episodes) + "_", save_fig=save_cur_fig)
        plot_results(ls2, output_folder, file_name="_2trans_" + str(num_episodes) + "_", save_fig=save_cur_fig)
        plot_results(ls3, output_folder, file_name="_3trans_" + str(num_episodes) + "_", save_fig=save_cur_fig)
        plot_results(ls4, output_folder, file_name="_4trans_" + str(num_episodes) + "_", save_fig=save_cur_fig)
        plot_results(ls5, output_folder, file_name="_5trans_" + str(num_episodes) + "_", save_fig=save_cur_fig)


def default_experiment(agent_name, env_name, num_of_epochs, num_of_episodes_per_epoch, num_states_in_partial_policy):
    # output_folder = "./output/"
    # if os.path.isdir(output_folder):
    #     shutil.rmtree(output_folder)
    # os.makedirs(output_folder)

    result = []

    for i in range(num_of_epochs):
        print('\nEpoch num:', i + 1)
        episode_result = run_experiment(env_name, agent_name, num_of_episodes_per_epoch, num_states_in_partial_policy)
        result.append(episode_result)

    # save result
    pkl_name = DATA_FOLDER + agent_name + '_all_stats_' + str(num_of_episodes_per_epoch) + "_states_" + str(
        num_states_in_partial_policy) + ".pkl"
    output = open(pkl_name, 'wb')
    pickle.dump(result, output)
    output.close()

    # plot
    # plot_results(result, output_folder)
    # plot_results_by_number_of_transforms(pkl_name, output_folder, num_of_episodes_per_epoch)
    return result


def different_envs_experiment():
    env_name = TAXI_EXAMPLE
    agent_name = Q_LEARNING
    num_of_episodes_per_epoch = 1000
    num_of_envs = 1
    num_states_in_partial_policy = 10
    all_env_results = {}
    all_env_test_results = {"original": [], "walls": [], "reward": [], "reward_walls": [], "fuel": [],
                            "fuel_walls": [], "fuel_reward": [], "fuel_reward_walls": []}
    all_env_evaluate_results = {"original": [], "walls": [], "reward": [], "reward_walls": [], "fuel": [],
                                "fuel_walls": [], "fuel_reward": [], "fuel_reward_walls": []}
    all_env_success_results = {"original": [], "walls": [], "reward": [], "reward_walls": [], "fuel": [],
                               "fuel_walls": [], "fuel_reward": [], "fuel_reward_walls": []}

    for i in range(num_of_envs):
        # set_up_env_idx()
        result = run_experiment(env_name, agent_name, num_of_episodes_per_epoch, num_states_in_partial_policy)
        all_env_results[i] = result
        for k, v in all_env_test_results.items():
            all_env_test_results[k].append(np.mean(result[k]['train_episode_reward_mean']))
            all_env_evaluate_results[k].append(np.mean(result[k]['evaluate_episode_reward_mean']))
            all_env_success_results[k].append(np.mean(result[k]['success_rate']))
        plot_result_graphs(agent_name, result)

        # visualize rewards and success_rate
    # names = ["original", "walls", "reward", "reward_walls", "fuel", "fuel_walls", "fuel_reward", "fuel_reward_walls"]
    # plot_graph_by_transform_name_and_env(agent_name, all_env_test_results, "300_env_test_results_5000_episodes")
    # plot_graph_by_transform_name_and_env(agent_name, all_env_evaluate_results, "300_env_evaluate_results_5000_episodes")
    # plot_success_rate_charts(names, [np.mean(v) for v in all_env_success_results.values()],
    #                          "success_rate_300_env_5000_episodes")


def run_anticipated_dfs_search():
    cur_env_preconditions = load_pkl_file(PRECONDITIONS_PATH)
    precondition_graph = load_pkl_file(PRECONDITION_GRAPH_PATH)
    if not precondition_graph:
        precondition_graph = PreconditionsGraph(SINGLE_TAXI_EXAMPLE, cur_env_preconditions.not_allowed_features,
                                                ANTICIPATED_POLICY)
        save_pkl_file("precondition_graph_small_taxi_env.pkl", precondition_graph)
    precondition_graph.bfs()


def create_transform_search_env():
    from TransformSearch.taxi_transform_search_env import create_transform_search_taxi_env, reformat_preconditions

    env = get_env(SINGLE_TAXI_EXAMPLE)
    preconditions = load_pkl_file(PRECONDITIONS_PATH)
    new_env_preconditions = reformat_preconditions(preconditions)
    new_env = create_transform_search_taxi_env(env, new_env_preconditions, ANTICIPATED_POLICY)
    return new_env


# if __name__ == '__main__':
#     from TransformSearch.greedy_search import greedy_search
#
#     preconditions = load_pkl_file(PRECONDITIONS_PATH)
#     max_transformed_env_opt, best_cluster = greedy_search(preconditions, ANTICIPATED_POLICY)
#     a = 7