from Environments.SingleTaxiEnv.single_taxi_env import *

SMALL_EXAMPLE_MAP = [
    "+-----+",
    "|R| :G|",
    "| : : |",
    "|Y:F|B|",
    "+-----+",
]


class SingleSmallTaxiEnv(SingleTaxiEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich
    Description:
    There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off, the episode ends.
    Observations:
    There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), and 4 destination locations.
    Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi
    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger
    - 6: refuel the taxi
    Rewards:
    There is a default per-step reward of -1,
    except for delivering the passenger, which is +20,
    or executing "pickup" and "drop-off" actions illegally, which is -10.
    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, Y and B): locations for passengers and destinations
    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, deterministic=True):
        self.init_state, self.init_row, self.init_col = 6152, 2, 0
        self.deterministic = deterministic
        self.desc = np.asarray(SMALL_EXAMPLE_MAP, dtype='c')
        w, h = self.desc.shape
        self.last_action = None
        self.passengers_locations, self.fuel_station = self.get_info_from_map()
        self.taxi_fuel = MAX_FUEL
        self.num_rows = int(w - 2)
        self.num_columns = int((h - 1) / 2)
        self.num_states = (self.num_rows * self.num_columns) * (len(self.passengers_locations) + 1) * (
            len(self.passengers_locations)) * MAX_FUEL
        self.max_row = self.num_rows - 1
        self.max_col = self.num_columns - 1
        self.initial_state_distribution = np.zeros(self.num_states)
        self.num_actions = len(ACTIONS)
        self.passenger_in_taxi = len(self.passengers_locations)
        self.P = self.build_transition_matrix(deterministic=deterministic)
        # self.initial_state_distribution /= self.initial_state_distribution.sum()
        discrete.DiscreteEnv.__init__(self, self.num_states, self.num_actions, self.P, self.initial_state_distribution)
