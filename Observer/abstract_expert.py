from abc import ABC, abstractmethod


class AbstractExpert(ABC):
    @abstractmethod
    def get_expert_policy_set(self, state_tuple):
        pass

    @abstractmethod
    def full_expert_policy_dict(self):
        pass
