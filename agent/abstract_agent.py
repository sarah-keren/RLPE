from abc import ABC, abstractmethod


class AbstractAgent(ABC):
    def __init__(self, env, timesteps_per_episode=10001):
        self.env = env
        self.timesteps_per_episode = timesteps_per_episode

    @abstractmethod
    def run(self) -> {str: float}:
        """
        The agent's training method.
        Returns: a dictionary - {"episode_reward_mean": __, "episode_reward_min": __, "episode_reward_max": __,
        "episode_len_mean": __}
        """
        pass

    @abstractmethod
    def compute_action(self, state) -> int:
        """
        Computes the best action from a given state.
        Returns: a int that represents the best action.
        """
        pass

    @abstractmethod
    def stop_episode(self):
        pass

    @abstractmethod
    def episode_callback(self, state, action, reward, next_state, terminated):
        pass

    @abstractmethod
    def evaluate(self):
        pass
