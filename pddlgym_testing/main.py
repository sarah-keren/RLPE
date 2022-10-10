import matplotlib  # matplotlib.use('agg')
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['SM_FRAMEWORK'] = "tf.keras"
from Agents.RL_agents import rl_agent
from Agents.RL_agents.q_learning_agents import Q_LEARNING
from constants import *
import pddlgym.pddlgym as pddlgym
from pddl_parser.planner import Planner
from run_utils import *

matplotlib.use('agg')  # For rendering


def save_data(name, data):
    output = open(name, 'wb')
    pickle.dump(data, output)
    output.close()


if __name__ == '__main__':
    # sokoban, Blocks World, Towers of Hanoi, Snake, Rearrangements, Triangle Tireworld and Exploding Blocks
    env_name = "sokoban"
    agent_name = Q_LEARNING
    num_of_epochs = 600000
    search_method = BASE
    run_rlpe(agent_name, env_name, search_method, num_of_epochs)