# ================= Train variables ================ #
ITER_NUM = 1000000
# =================== Transforms =================== #
FUELS_TRANSFORM = "fuels_transform"
REWARD_TRANSFORM = "reward_transform"
NO_WALLS_TRANSFORM = "no_walls_transform"
WITHOUT_TRANSFORM = "Without transform"
# ================== Environments ================== #
TAXI = "taxi"
SPEAKER_LISTENER = "simple_speaker_listener"
TAXI_STATES_NUM = 5000
SEARCH_TRANSFORM_TAXI_ENV = "big_taxi_search_transform_env"
APPLE_PICKING = "apple_picking_env"
# ================= Env variables ================== #

TRAIN = "train"
EVALUATE = "evaluate"
EPISODE_REWARD_MEAN = "episode_reward_mean"
EPISODE_REWARD_MAX = "episode_reward_max"
EPISODE_REWARD_MIN = "episode_reward_min"
EPISODE_STEP_NUM_MEAN = "episode_len_mean"
EPISODE_VARIANCE = "episode_variance"
TOTAL_EPISODE_REWARD = "total_episode_reward"
TRAINING_RESULTS = "training_results"
EVALUATION_RESULTS = "evaluation_results"
SUCCESS_RATE = "success_rate"
GOT_AN_EXPLANATION = "explanation"
ORIGINAL_ENV = "original_env"
FUEL_TRANSFORMED_ENV = "fuel"

SMALL_TAXI_ANTICIPATED_POLICY = {(2, 0, 0, 3, None): [1],
                                 (1, 0, 0, 3, None): [1],
                                 (0, 0, 0, 3, None): [4],
                                 (0, 0, 4, 3, None): [0],
                                 (1, 0, 4, 3, None): [2],
                                 (1, 1, 4, 3, None): [2],
                                 (1, 2, 4, 3, None): [0],
                                 (2, 2, 4, 3, None): [5]}

BIG_TAXI_ANTICIPATED_POLICY = {(4, 0, 0, 3, None): [1],
                               (3, 0, 0, 3, None): [1],
                               (2, 0, 0, 3, None): [1],
                               (1, 0, 0, 3, None): [1],
                               (0, 0, 0, 3, None): [4],
                               (0, 0, 4, 3, None): [0, 2],
                               (0, 1, 4, 3, None): [0],
                               (1, 0, 4, 3, None): [0, 2],
                               (1, 1, 4, 3, None): [0],
                               (2, 0, 4, 3, None): [2],
                               (2, 1, 4, 3, None): [2],
                               (2, 2, 4, 3, None): [2],
                               (2, 3, 4, 3, None): [0, 2],
                               (2, 4, 4, 3, None): [0],
                               (3, 3, 4, 3, None): [0, 2],
                               (3, 4, 4, 3, None): [0],
                               (4, 3, 4, 3, None): [2],
                               (4, 4, 4, 3, None): [5]
                               }
ANTICIPATED_POLICY = BIG_TAXI_ANTICIPATED_POLICY

PRECONDITION_NAME_IDX = 0
NEXT_STATE_IDX = 1
# ---------------------- PATHS ----------------------- #
CUR_ENV_NAME = "big_taxi"

DATA_FOLDER = "Data/"
TRAINED_AGENTS_DIR_PATH = DATA_FOLDER + "TrainedAgentsDQN/"
TRAINED_AGENT_SAVE_PATH = TRAINED_AGENTS_DIR_PATH + f"trained_models_on_{CUR_ENV_NAME}_env/"
TRAINED_AGENT_SPECIFIC_PATH = "single_transform_envs/"
TRAINED_AGENT_ON_SPECIFIC_TRANSFORM_NUM_PATH = TRAINED_AGENT_SAVE_PATH + TRAINED_AGENT_SPECIFIC_PATH
TRAINED_AGENT_RESULTS_PATH = TRAINED_AGENTS_DIR_PATH + f"results/trained_models_on_{CUR_ENV_NAME}_env_results/"
TRAINED_AGENT_RESULT_FILE_PATH = TRAINED_AGENT_RESULTS_PATH + f"results_{CUR_ENV_NAME}_search_transform.pkl"

TAXI_TRANSFORM_DATA_PATH = DATA_FOLDER + f"{CUR_ENV_NAME}_env_data/"
PRECONDITIONS_FILE_NAME = f"{CUR_ENV_NAME}_env_preconditions.pkl"
PRECONDITIONS_PATH = TAXI_TRANSFORM_DATA_PATH + PRECONDITIONS_FILE_NAME
PRECONDITION_GRAPH_PATH = TAXI_TRANSFORM_DATA_PATH + f"precondition_graph_{CUR_ENV_NAME}_env.pkl"
ALL_SMALL_TAXI_TRANSFORMED_ENVS_PATH = TAXI_TRANSFORM_DATA_PATH + "all_single_transformed_envs.pkl"
TRANSFORMS_PATH = TAXI_TRANSFORM_DATA_PATH + f"{CUR_ENV_NAME}_transformed_envs/"

TRANSFORM_SEARCH_TAXI_ENV_FILENAME = f"{CUR_ENV_NAME}_search_transform_env.pkl"
TRANSFORM_SEARCH_TAXI_ENV_PATH = TAXI_TRANSFORM_DATA_PATH + TRANSFORM_SEARCH_TAXI_ENV_FILENAME

PATH_TO_DATA_FOR_COLAB = DATA_FOLDER + "collab_example_data/"
