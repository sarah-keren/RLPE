SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF, REFUEL = 0, 1, 2, 3, 4, 5, 6
# PASSENGER_IN_TAXI = -1
STEP_REWARD, PICKUP_REWARD, BAD_PICKUP_REWARD, DROPOFF_REWARD, BAD_DROPOFF_REWARD, REFUEL_REWARD, BAD_REFUEL_REWARD, NO_FUEL_REWARD = "step", "good_pickup", "bad_pickup", "good_dropoff", "bad_dropoff", "good_refuel", "bad_refuel", "no_fuel"
ACTIONS = [SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF, REFUEL]
MAX_FUEL = 50
REWARD_DICT = {STEP_REWARD: -1,
               PICKUP_REWARD: 0, BAD_PICKUP_REWARD: -10,
               DROPOFF_REWARD: 100, BAD_DROPOFF_REWARD: -10,
               REFUEL_REWARD: 0, BAD_REFUEL_REWARD: -10, NO_FUEL_REWARD: -100}

DETERMINISTIC_PROB = 1.0
STOCHASTIC_PROB = 0.91
STOCHASTIC_PROB_OTHER_ACTIONS = (DETERMINISTIC_PROB - STOCHASTIC_PROB) / 3
