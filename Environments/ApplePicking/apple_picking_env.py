import copy
import math
import sys
import numpy as np
from contextlib import closing
from io import StringIO
from gym import utils
from gym.envs.toy_text import discrete
from Environments.ApplePicking.apple_picking_constants import *

# MAP = [
#     "+---------+",
#     "|A: |z| :A|",
#     "| : : : : |",
#     "| : : : : |",
#     "| : : : : |",
#     "|S|z|A|z|A|",
#     "+---------+",
# ]

MAP = [
    "+---------+",
    "|A: : : :A|",
    "| : : : : |",
    "| : : : : |",
    "| : : : : |",
    "|S: :A: :A|",
    "+---------+",
]


# MAP = [
#     "+-----+",
#     "|A: :A|",
#     "| : :A|",
#     "|S: :A|",
#     "+-----+",
# ]


def try_step_south_or_east(place, max_place):
    new_place = min(place + 1, max_place)
    return new_place


def try_step_west_or_north(place, max_place):
    new_place = max(place - 1, max_place)
    return new_place


# def apples_valid(a1, a2, a3, a4):
#     result = not (
#             a1 != 0 and a1 == a2 and a1 == a3 and a1 == a4 and a2 != 0 and a2 == a3 and a2 == a4 and a3 != 0 and a3 == a4)
#     tamp_apple_arr = [a1, a2, a3, a4]
#     max_apple = max([a1, a2, a3, a4])
#     for i in range(1, max_apple):
#         if i not in tamp_apple_arr:
#             result = False
#     return result


class ApplePickingEnv(discrete.DiscreteEnv):
    """
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup an apple
    Rewards:
    - (-1): step
    - (-3): entering thorny area
    - (-10): excecute "pick
    Rendering:
    - Purple A: not collected apple
    - White A: collected apple
    - Green z: thorn
    - Blue: the collector
    state space is represented by:
        (collector_row, collector_col, apple1_location, apple2_location, apple3_location, apple4_location)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')
        w, h = self.desc.shape
        self.last_action = None
        self.apple_locations, self.thorny_locations, self.start_position = self.get_info_from_map()
        self.original_apple_locations = copy.deepcopy(self.apple_locations)
        self.num_of_picked_apples = 0
        self.num_rows = int(w - 2)
        self.num_columns = int((h - 1) / 2)
        self.num_states = (self.num_rows * self.num_columns) * int(math.pow(2, len(self.apple_locations)))
        self.max_row = self.num_rows - 1
        self.max_col = self.num_columns - 1
        self.initial_state_distribution = np.zeros(self.num_states)
        self.num_actions = len(ACTIONS)
        self.P = self.build_transition_matrix()
        # self.initial_state_distribution /= self.initial_state_distribution.sum()
        discrete.DiscreteEnv.__init__(self, self.num_states, self.num_actions, self.P, self.initial_state_distribution)

    def build_transition_matrix(self):
        """
        Build the transition matrix.
        You can work with deterministic environment or stochastic one by changing the flag in the arguments.
        return: dictionary with the transition matrix
        """
        P = {state: {action: [] for action in range(self.num_actions)} for state in range(self.num_states)}
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                for apple1 in range(2):
                    for apple2 in range(2):
                        for apple3 in range(2):
                            for apple4 in range(2):
                                state = self.encode(row, col, apple1, apple2, apple3, apple4)
                                apple_arr = [apple1, apple2, apple3, apple4]
                                num_of_picked_apples = sum(apple_arr)
                                if row == self.start_position[0] and col == self.start_position[
                                    1] and num_of_picked_apples == 0:
                                    self.initial_state_distribution[state] += 1.0
                                for action in range(self.num_actions):
                                    new_row, new_col = row, col
                                    new_apple1, new_apple2, new_apple3, new_apple4 = apple1, apple2, apple3, apple4
                                    new_apple_arr = [new_apple1, new_apple2, new_apple3, new_apple4]
                                    reward = REWARD_DICT[STEP_REWARD]  # default reward when there is no pickup
                                    done = True if num_of_picked_apples == 4 else False
                                    collector_loc = (row, col)

                                    if action in [SOUTH, NORTH, WEST, EAST]:
                                        new_row, new_col, reward = self.try_to_move(action, row, col)

                                    elif action == PICKUP:
                                        new_apple_arr, reward, done = self.try_picking_up(collector_loc, done,
                                                                                          new_apple_arr, reward)
                                    self.num_of_picked_apples = num_of_picked_apples
                                    new_state = self.encode(new_row, new_col, *new_apple_arr)
                                    if self.near_thorny_wall(row, col, ABOVE):
                                        P[state][action] = self.get_stochastic_probs(action, row, col, apple_arr,
                                                                                     new_state, reward, done, ABOVE)
                                    elif self.near_thorny_wall(row, col, UNDER):
                                        P[state][action] = self.get_stochastic_probs(action, row, col, apple_arr,
                                                                                     new_state, reward, done, UNDER)
                                    else:
                                        P[state][action] = self.get_stochastic_probs(action, row, col, apple_arr,
                                                                                     new_state, reward, done)
        return P

    def encode(self, collector_row, collector_col, apple1, apple2, apple3, apple4):
        # (5), 5, 5, 5, 5, 5
        # (num_rows), num_columns, 2 X 4
        i = collector_row
        i *= self.num_columns
        i += collector_col
        i *= 2
        i += apple1
        i *= 2
        i += apple2
        i *= 2
        i += apple3
        i *= 2
        i += apple4
        return i

    def decode(self, i):
        # 2 X 4, 5, (5)
        # 2 X 4, num_columns, (num_rows)
        out = []
        out.append(i % 2)
        i = i // 2
        out.append(i % 2)
        i = i // 2
        out.append(i % 2)
        i = i // 2
        out.append(i % 2)
        i = i // 2
        out.append(i % self.num_columns)
        i = i // self.num_columns
        out.append(i)
        assert 0 <= i < self.num_rows
        return list(reversed(out))

    def near_thorny_wall(self, new_row, new_col, value):
        for thorny_wall in self.thorny_locations:
            if new_row == thorny_wall[0] + value and new_col == thorny_wall[1]:
                return True
        return False

    def check_if_state_is_legal(self, state, return_idxes=False):
        if isinstance(state, int):
            collector_row, collector_col, apple1, apple2, apple3, apple4 = self.decode(state)
        else:
            collector_row, collector_col, apple1, apple2, apple3, apple4 = state
        not_valid_idx = []
        if collector_row < 0 or collector_row >= self.num_rows:
            not_valid_idx.append(0)
        if collector_col < 0 or collector_col >= self.num_columns:
            not_valid_idx.append(1)
        if apple_locations < 0 or apple_locations > len(self.apple_locations):  # TODO - bug!!!
            not_valid_idx.append(2)
        state_is_legal = (len(not_valid_idx) == 0)
        if return_idxes:
            return state_is_legal, not_valid_idx
        return state_is_legal

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        collector_row, collector_col, apple1, apple2, apple3, apple4 = self.decode(self.s)
        apple_arr = apple1, apple2, apple3, apple4

        def ul(x):
            return "_" if x == " " else x

        def colorize_loc(j, cur_color, apple_arr):
            if apple_arr[j] == NOT_COLLECTED:
                xc, yc = self.apple_locations[j]
                out[1 + xc][2 * yc + 1] = utils.colorize(out[1 + xc][2 * yc + 1], cur_color)

        out[1 + collector_row][2 * collector_col + 1] = utils.colorize(out[1 + collector_row][2 * collector_col + 1],
                                                                       'blue', highlight=True)

        for (di, dj) in self.thorny_locations:
            out[1 + di][2 * dj + 1] = utils.colorize(ul(out[1 + di][2 * dj + 1]), 'green', bold=True)

        # colors = ['magenta', 'cyan', 'crimson', 'yellow', 'red', 'white']
        for j in range(len(self.apple_locations)):
            colorize_loc(j, 'magenta', apple_arr)

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup"][self.lastaction]))
        else:
            outfile.write("\n")
        print("current state: ", self.decode(self.s), ", last action: ", self.last_action)
        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

    def reset(self):
        self.s = discrete.categorical_sample(self.isd, self.np_random)
        self.last_action = None
        return int(self.s)

    def step(self, a):
        transitions = self.P[self.s][a]
        i = discrete.categorical_sample([t[0] if len(t) > 0 else 0 for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.last_action = a
        return int(s), r, d, {"prob": p}

    def try_to_move(self, action, row, col):
        new_row, new_col = row, col
        if action == SOUTH:
            new_row = try_step_south_or_east(row, self.max_row)
        elif action == NORTH:
            new_row = try_step_west_or_north(row, 0)
        elif action == EAST and self.no_wall_to_the_right(row, col):
            new_col = try_step_south_or_east(col, self.max_col)
        elif action == WEST and self.no_wall_to_the_left(row, col):
            new_col = try_step_west_or_north(col, 0)
        reward = REWARD_DICT[STEP_INTO_THORN_REWARD] if (new_row, new_col) in self.thorny_locations else REWARD_DICT[
            STEP_REWARD]
        return new_row, new_col, reward

    def get_stochastic_probs(self, action, row, col, apple_arr, new_state, reward, done, near_thorny=0):
        # return [(DETERMINISTIC_PROB, new_state, reward, done)]
        if near_thorny == 0:
            return self.prob_list_for_no_move_action(action, new_state, reward, done)
        risky_action = None
        prob_list = [tuple() for _ in range(self.num_actions)]
        if near_thorny == ABOVE:
            risky_action = SOUTH
        elif near_thorny == UNDER:
            risky_action = NORTH
        action_prob = (STOCHASTIC_PROB, new_state, reward, done)
        prob_list[action] = action_prob
        for prob_act in range(len(prob_list)):
            if prob_act != action:
                if prob_act == risky_action:
                    new_row, new_col, reward = self.try_to_move(prob_act, row, col)
                    done = False
                    new_state = self.encode(new_row, new_col, *apple_arr)
                    prob_list[prob_act] = (STOCHASTIC_PROB_THORNY, new_state, reward, done)
                else:
                    prob_list[prob_act] = (0.0, new_state, reward, done)
            elif action == risky_action:
                prob_list[prob_act] = (1.0, new_state, reward, done)
        return prob_list

    def prob_list_for_no_move_action(self, action, new_state, reward, done):
        prob_list = [tuple() for _ in range(self.num_actions)]
        prob_list[action] = (DETERMINISTIC_PROB, new_state, reward, done)
        for a in range(len(prob_list)):
            if a != action:
                prob_list[a] = (0.0, new_state, reward, done)
        return prob_list

    def no_wall_to_the_right(self, row, col):
        return self.desc[1 + row, 2 * col + 2] == b":"

    def no_wall_to_the_left(self, row, col):
        return self.desc[1 + row, 2 * col] == b":"

    def try_picking_up(self, collector_loc, done, new_apple_arr, reward):
        for i, apple_loc in enumerate(self.apple_locations):
            if collector_loc == apple_loc and new_apple_arr[i] == NOT_COLLECTED:
                old_apple_arr = copy.deepcopy(new_apple_arr)
                new_apple_arr[i] = COLLECTED
                if sum(new_apple_arr) == 4 and sum(old_apple_arr) != 4:
                    done = True
                reward = REWARD_DICT[APPLE_PICKUP_REWARD]
                break
        if reward != REWARD_DICT[APPLE_PICKUP_REWARD]:
            reward = REWARD_DICT[BAD_APPLE_PICKUP_REWARD]
        return new_apple_arr, reward, done

    def get_info_from_map(self):
        apple_locations, thorny_wall_locations, start_position = [], [], None
        h, w = self.desc.shape
        h, w = (h - 2), (w - 2)
        for x in range(1, h + 1):
            for y in range(1, w + 1):
                c = self.desc[x][y]
                if c == b'A':
                    apple_locations.append((x - 1, int(y / 2)))
                elif c == b'z':
                    thorny_wall_locations.append((x - 1, int(y / 2)))
                elif c == b'S':
                    start_position = (x - 1, int(y / 2))

        return apple_locations, thorny_wall_locations, start_position


# if __name__ == '__main__':
#     new_env = ApplePickingEnv()
#     new_env.reset()
#     actions = [1, 1, 1, 1, 4, 2, 0, 0, 2, 2, 1, 1, 2, 4, 0, 0, 0, 0, 4, 1, 1, 3, 3, 0, 0, 4, 1]
#     actions = [1, 1, 1, 1, 4, 2, 0, 2, 2, 1, 1, 2, 4, 0, 0, 0, 0, 4, 1, 1, 3, 3, 0, 0, 4, 1]
#     all_reward = 0
#     for act in actions:
#         new_env.render()
#         next_s, r, d, prob = new_env.step(act)
#         all_reward += r
#         print("state:", new_env.decode(next_s))
#         print(f"{next_s}, {r}, {d}, {prob}")
#         print("all_reward:", all_reward)
#         if d:
#             break
