from experiments import *
from Transforms.single_taxi_transforms import *
from save_load_utils import *



def main():
    # define the environment
    env_name = SINGLE_TAXI_EXAMPLE
    agent_name = KERAS_DQN
    num_of_episodes = 200000
    results = {}
    original_env = get_env(env_name)

    # anticipated policy is <state, action> pairs
    anticipated_policy = ANTICIPATED_POLICY

    new_agent = load_existing_agent(original_env, agent_name, ORIGINAL_ENV, TRAINED_AGENTS_DIR_PATH)

    evaluation_result = rl_agent.run(new_agent, 5, method=EVALUATE, print_process=False, visualize=False)

    anticipated_policy_achieved, success_rate = is_anticipated_policy_achieved(original_env, new_agent,
                                                                               anticipated_policy)
    if anticipated_policy_achieved:
        print("The algorithm achieved the policy. We finished our work.")

    # transforms = load_existing_transforms_from_dir()
    transform_list = os.listdir(TRANSFORMS_PATH)
    # cur_transform_name = "0_(4,)_[0]_1_(4,)_[0]_2_(4,)_[0]_4_(4,)_[0]_5_(4,)_[0].pkl"
    # t_name, t_env = load_transform_by_name(cur_transform_name, dir_name=TRANSFORMS_PATH)
    # transforms = {0: (t_name, t_env)}
    explanations = []
    for f in transform_list:
        # for params, (transform_name, transformed_env) in transforms.items():
        # create agent
        transform_name, transformed_env = load_transform_by_name(f)
        print(f"\nEvaluating agent on the transformed environment: {transform_name}")
        agent = load_existing_agent(transformed_env, agent_name, transform_name, TRAINED_AGENTS_DIR_PATH)
        if agent is None:
            continue
        transformed_train_result, explanation = load_existing_results(agent_name, transform_name, num_of_episodes)
        # evaluate the performance of the agent
        transformed_evaluation_result = rl_agent.run(agent, num_of_episodes, method=EVALUATE, print_process=True,
                                                     visualize=False)

        # check if the anticipated policy is achieved in trans_env
        anticipated_policy_achieved, success_rate = is_anticipated_policy_achieved(original_env, agent,
                                                                                   anticipated_policy)

        results[transform_name] = {EVALUATION_RESULTS: transformed_evaluation_result,
                                   TRAINING_RESULTS: transformed_train_result,
                                   SUCCESS_RATE: success_rate,
                                   GOT_AN_EXPLANATION: False}
        if anticipated_policy_achieved:
            print(f"Got an explanation: {transform_name}")
            explanations.append(transform_name)
            results[transform_name][GOT_AN_EXPLANATION] = True

    if explanations is None or len(explanations) == 0:
        print("No explanation found! :-(")
    else:
        print(f"Explanations found: {explanations}")

    success_rates = [v[SUCCESS_RATE] for v in results.values()]
    names = [_ for _ in range(len(success_rates))]
    names_translation_dict = dict()
    for i, k in enumerate(results.keys()):
        names_translation_dict[i] = k
    fig, ax = plt.subplots()
    ax.bar(names, success_rates)
    ax.set_xticks(np.arange(len(names)))
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate of the different transformed environments")
    print(f"for translating the labels into transform name: \n{names_translation_dict}")
    plt.show()
    a = 7

# main()
