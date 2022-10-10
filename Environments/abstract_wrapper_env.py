from abc import ABC, abstractmethod


class AbstractWrapperEnv(ABC):
    """
    This is the basic AbstractTransformedEnvironment requirements for working with this project.
    """

    @abstractmethod
    def get_states_from_partial_obs(self, partial_obs):
        """
        ...
        """
        pass

    @abstractmethod
    def encode(self, state):
        """
        ...
        """
        pass

    @abstractmethod
    def decode(self, i):
        pass

