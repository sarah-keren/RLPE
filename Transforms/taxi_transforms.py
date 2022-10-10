import itertools
from Environments.MultiTaxiEnv.multitaxienv.config import TAXI_ENVIRONMENT_REWARDS
from Environments.taxi_environment_wrapper import TaxiSimpleExampleEnv
from constants import *
from Transforms.transform_constants import *

SMALL_MAP = [
    "+-----+",
    "|X: :X|",
    "| : : |",
    "|X:F:X|",
    "+-----+",
]

# NEW_MAP = [
#     "+---------+",
#     "|X: :F: :X|",
#     "| : : : : |",
#     "| : : : : |",
#     "| : : : : |",
#     "|X: :G:X: |",
#     "+---------+",
# ]
TAXI_NAME = "taxi_1"
my_reward = dict(
    step=-1,
    no_fuel=-20,
    bad_pickup=-15,
    bad_dropoff=-15,
    bad_refuel=-10,
    bad_fuel=-50,
    pickup=50,
    standby_engine_off=-1,
    turn_engine_on=-10e6,
    turn_engine_off=-10e6,
    standby_engine_on=-1,
    intermediate_dropoff=-15,
    final_dropoff=100,
    hit_wall=-2,
    collision=-35,
    collided=-20,
    unrelated_action=-15
)


# fuel_transform=False, status_transform=False, reward_transform=False

class TaxiTransformedEnv(TaxiSimpleExampleEnv):
    def __init__(self, transforms):
        super().__init__(max_fuel=[3] if transforms[2] else [100])
        taxi_loc_x_transform, taxi_loc_y_transform = transforms[0], transforms[1]
        fuel_transform = transforms[2]
        pass_loc_x_transform, pass_loc_y_transform = transforms[3], transforms[4]
        pass_dest_x_transform, pass_dest_y_transform = transforms[5], transforms[6]
        pass_status_transform = transforms[7]
        no_walls_transform = transforms[8]
        reward_transform = transforms[9]

        self.taxi_loc_x_transform = taxi_loc_x_transform
        self.taxi_loc_y_transform = taxi_loc_y_transform
        self.fuel_transform = fuel_transform
        self.pass_loc_x_transform = pass_loc_x_transform
        self.pass_loc_y_transform = pass_loc_y_transform
        self.pass_dest_x_transform = pass_dest_x_transform
        self.pass_dest_y_transform = pass_dest_y_transform
        self.pass_status_transform = pass_status_transform
        self.no_walls_transform = no_walls_transform
        self.reward_transform = reward_transform
        self.reward_dict = my_reward if self.reward_transform else TAXI_ENVIRONMENT_REWARDS

    def _update_movement_wrt_fuel(self, taxi: int, taxis_locations: list, wanted_row: int, wanted_col: int,
                                  reward: int, fuel: int) -> (int, int, list):
        """
        Given that a taxi would like to move - check the fuel accordingly and update reward and location.
        Args:
            taxi: index of the taxi
            taxis_locations: list of current locations (prior to movement)
            wanted_row: row after movement
            wanted_col: col after movement
            reward: current reward
            fuel: current fuel

        Returns: updated_reward, updated fuel, updared_taxis_locations

        """
        reward = reward
        fuel = fuel
        taxis_locations = taxis_locations
        if fuel == 0:
            reward = ('no_fuel')
        else:
            if not self.fuel_transform:
                fuel = max(0, fuel - 1)

            taxis_locations[taxi] = [wanted_row, wanted_col]

        return reward, fuel, taxis_locations

    def set_reward_dict(self, new_rewards):
        self.reward_dict = new_rewards

    def step(self, action):
        """
        Executing a list of actions (action for each taxi) at the domain current state.
        Supports not-joined actions, just pass 1 element instead of list.
        Using manual reward.

        Args:
            action_dict: {taxi_name: action} - action of specific taxis to take on the step

        Returns: - dict{taxi_id: observation}, dict{taxi_id: reward}, dict{taxi_id: done}, _
        """
        action_dict = {TAXI_NAME: action}
        rewards = {}
        self.counter += 1
        if self.counter >= 90:
            self.window_size = 3

        # Main of the function, for each taxi-i act on action[i]
        for taxi_name, action_list in action_dict.items():
            # meta operations on the type of the action
            action_list = self._get_action_list(action_list)

            for action in action_list:
                taxi = self.taxis_names.index(taxi_name)
                reward = self.partial_closest_path_reward('step')  # Default reward
                moved = False  # Indicator variable for later use

                # taxi locations: [i, j]
                # fuels: int
                # passengers_start_locations and destinations: [[i, j] ... [i, j]]
                # passengers_status: [[1, 2, taxi_index+2] ... [1, 2, taxi_index+2]], 1 - delivered
                taxis_locations, fuels, passengers_start_locations, destinations, passengers_status = self.state

                if all(list(self.dones.values())):
                    continue

                # If taxi is collided, it can't perform a step
                if self.collided[taxi] == 1:
                    rewards[taxi_name] = self.partial_closest_path_reward('collided')
                    self.dones[taxi_name] = True
                    continue

                # If the taxi is out of fuel, it can't perform a step
                if fuels[taxi] == 0 and not self.at_valid_fuel_station(taxi, taxis_locations):
                    rewards[taxi_name] = self.partial_closest_path_reward('bad_fuel')
                    self.dones[taxi_name] = True
                    continue

                taxi_location = taxis_locations[taxi]
                row, col = taxi_location

                fuel = fuels[taxi]
                is_taxi_engine_on = self.engine_status_list[taxi]
                _, index_action_dictionary = self.get_available_actions_dictionary()

                if not is_taxi_engine_on:  # Engine is off
                    # update reward according to standby/ turn-on/ unrelated + turn engine on if requsted
                    reward = self._engine_is_off_actions(index_action_dictionary[action], taxi)

                else:  # Engine is on
                    # Binding
                    if index_action_dictionary[action] == 'bind':
                        self.bounded = False
                        reward = self.partial_closest_path_reward('bind')

                    # Movement
                    if index_action_dictionary[action] in ['south', 'north', 'east', 'west']:
                        moved, row, col = self._take_movement(index_action_dictionary[action], row, col)

                    # Check for collisions
                    if self.collision_sensitive_domain and moved:
                        if self.collided[taxi] == 0:
                            reward, moved, action, taxis_locations = self._check_action_for_collision(taxi,
                                                                                                      taxis_locations,
                                                                                                      row, col, moved,
                                                                                                      action, reward)

                    # Pickup
                    elif index_action_dictionary[action] == 'pickup':
                        passengers_status, reward = self._make_pickup(taxi, passengers_start_locations,
                                                                      passengers_status, taxi_location, reward)

                    # Dropoff
                    elif index_action_dictionary[action] == 'dropoff':
                        passengers_status, passengers_start_locations, reward = self._make_dropoff(taxi,
                                                                                                   passengers_start_locations,
                                                                                                   passengers_status,
                                                                                                   destinations,
                                                                                                   taxi_location,
                                                                                                   reward)

                    # Turning engine off
                    elif index_action_dictionary[action] == 'turn_engine_off':
                        reward = self.partial_closest_path_reward('turn_engine_off')
                        self.engine_status_list[taxi] = 0

                    # Standby with engine on
                    elif index_action_dictionary[action] == 'standby':
                        reward = self.partial_closest_path_reward('standby_engine_on')

                # Here we have finished checking for action for taxi-i
                # Fuel consumption
                if moved:
                    reward, fuels[taxi], taxis_locations = self._update_movement_wrt_fuel(taxi, taxis_locations,
                                                                                          row, col, reward, fuel)

                if (not moved) and action in [self.action_index_dictionary[direction] for
                                              direction in ['north', 'south', 'west', 'east']]:
                    reward = self.reward_dict['hit_wall']

                # taxi refuel
                if index_action_dictionary[action] == 'refuel':
                    reward, fuels[taxi] = self._refuel_taxi(fuel, reward, taxi, taxis_locations)

                # check if all the passengers are at their destinations
                done = all(loc == 1 for loc in passengers_status)
                self.dones[taxi_name] = done

                # check if all taxis collided
                done = all(self.collided == 1)
                self.dones[taxi_name] = self.dones[taxi_name] or done

                # check if all taxis are out of fuel
                done = fuels[taxi] == 0
                self.dones[taxi_name] = self.dones[taxi_name] or done

                rewards[taxi_name] = reward
                self.state = [taxis_locations, fuels, passengers_start_locations, destinations, passengers_status]
                self.last_action = action_dict

        self.dones['__all__'] = True
        self.dones['__all__'] = all(list(self.dones.values()))

        if self.bounded:
            total_reward = 0
            for taxi_id in action_dict.keys():
                total_reward += rewards[taxi_id]
            total_reward /= len(action_dict.keys())
            for taxi_id in action_dict.keys():
                rewards[taxi_id] = total_reward

        obs = {}
        for taxi_id in action_dict.keys():
            obs[taxi_id] = self.get_observation(self.state, taxi_id)
        obs, reward, done = obs[TAXI_NAME][0], {taxi_id: rewards[taxi_id] for taxi_id in action_dict.keys()}[
            TAXI_NAME], self.dones[TAXI_NAME]
        if self.taxi_loc_x_transform:
            obs[0] = 0
        if self.taxi_loc_y_transform:
            obs[1] = 0
        if self.pass_loc_x_transform:
            obs[3] = 4
        if self.pass_loc_y_transform:
            obs[4] = 0
        if self.pass_dest_x_transform:
            obs[5] = 4
        if self.pass_dest_y_transform:
            obs[6] = 4
        if self.pass_status_transform:
            obs[7] = 3
        obs = self.encode(obs)
        return obs, reward, done, {}

    def partial_closest_path_reward(self, basic_reward_str: str, taxi_index: int = None) -> int:
        """
        Computes the reward for a taxi and it's defined by:
        dropoff[s] - gets the reward equal to the closest path multiply by 15, if the drive got a passenger further
        away - negative.
        other actions - basic reward from config table
        Args:
            basic_reward_str: the reward we would like to give
            taxi_index: index of the specific taxi

        Returns: updated reward

        """
        if basic_reward_str not in ['intermediate_dropoff', 'final_dropoff'] or taxi_index is None:
            return self.reward_dict[basic_reward_str]

        # [taxis_locations, fuels, passengers_start_locations, destinations, passengers_status]
        current_state = self.state
        passengers_start_locations = current_state[2]

        taxis_locations = current_state[0]

        passengers_status = current_state[-1]
        passenger_index = passengers_status.index(taxi_index + 3)
        passenger_start_row, passenger_start_col = passengers_start_locations[passenger_index]
        taxi_current_row, taxi_currrent_col = taxis_locations[taxi_index]

        return 15 * (self.passenger_destination_l1_distance(passenger_index, passenger_start_row, passenger_start_col) -
                     self.passenger_destination_l1_distance(passenger_index, taxi_current_row, taxi_currrent_col))

    def get_explanation(self):
        explanation = []
        if self.fuel_transform:
            explanation.append(FUELS_TRANSFORM)
        # if self.status_transform:
        #     explanation.append(STATUS_TRANSFORM)
        if self.reward_transform:
            explanation.append(REWARD_TRANSFORM)
        return explanation

    def _take_movement(self, action: str, row: int, col: int) -> (bool, int, int):
        """
        Takes a movement with regard to a specific location of a taxi,
        can take action even though there a wall in the direction of the action.
        Args:
            action: direction to move
            row: current row
            col: current col

        Returns: if moved (always true), new row, new col

        """
        moved = False
        new_row, new_col = row, col
        max_row = self.num_rows - 1
        max_col = self.num_columns - 1
        if action == 'south':  # south
            if row != max_row:
                moved = True
            new_row = min(row + 1, max_row)
        elif action == 'north':  # north
            if row != 0:
                moved = True
            new_row = max(row - 1, 0)
        if action == 'east' and (self.no_walls_transform or self.desc[1 + row, 2 * col + 2] == b":"):  # east
            if col != max_col:
                moved = True
            new_col = min(col + 1, max_col)
        elif action == 'west' and (self.no_walls_transform or self.desc[1 + row, 2 * col] == b":"):  # west
            if col != 0:
                moved = True
            new_col = max(col - 1, 0)

        return moved, new_row, new_col

    #
    # TAXI_TRANSFORM_LIST = [TAXI_LOC_X, TAXI_LOC_Y, FUEL, PASS_LOC_X, PASS_LOC_Y, PASS_DEST_X, PASS_DEST_Y,
    # PASS_STATUS, WALLS, REWARD]


def get_taxi_transform_name(transforms):
    taxi_loc_x_transform, taxi_loc_y_transform = transforms[0], transforms[1]
    fuel_transform = transforms[2]
    pass_loc_x_transform, pass_loc_y_transform = transforms[3], transforms[4]
    pass_dest_x_transform, pass_dest_y_transform = transforms[5], transforms[6]
    pass_status_transform = transforms[7]
    no_walls_transform = transforms[8]
    reward_transform = transforms[9]
    name = ""
    name += TAXI_LOC_X if taxi_loc_x_transform else ""
    name += TAXI_LOC_Y if taxi_loc_y_transform else ""
    name += FUEL if fuel_transform else ""
    name += PASS_LOC_X if pass_loc_x_transform else ""
    name += PASS_LOC_Y if pass_loc_y_transform else ""
    name += PASS_DEST_X if pass_dest_x_transform else ""
    name += PASS_DEST_Y if pass_dest_y_transform else ""
    name += PASS_STATUS if pass_status_transform else ""
    name += WALLS if no_walls_transform else ""
    name += REWARD if reward_transform else ""
    return name
