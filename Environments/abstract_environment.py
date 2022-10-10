from abc import ABC, abstractmethod


class AbstractEnvironment(ABC):
    """
    This is the basic environment requirements for working with this project.
    """
    @abstractmethod
    def reset(self):
        """
          Resets the current state to the start state
        """
        pass

    @abstractmethod
    def render(self):
        """
        Renders the environment.
        """
        pass

    @abstractmethod
    def step(self, action):
        """
          Performs the given action in the current
          environment state and updates the environment.

          Returns (new_obs, reward, done, info)
        """
        pass
