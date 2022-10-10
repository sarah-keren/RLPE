class Agent():

    # init agents and their observations
    def __init__(self, decision_maker, observation):
        self.decision_maker = decision_maker
        self.observation = observation

    def get_action(self, environment, cur_observation):
        return self.decision_maker.get_decision(environment, cur_observation)

    def get_observation(self, state, param):
        return self.observation(state, param)