import abc

class Model(abc.ABC):
    """
    Base model class
    """
    @abc.abstractmethod
    def observation_space(self):
        pass

    @abc.abstractmethod
    def action_space(self):
        pass

    @abc.abstractmethod
    def predict(self, state, action):
        pass

    @abc.abstractmethod
    def rollout(self, state, actions):
        pass

    @abc.abstractmethod
    def get_observations(self, batch):
        """
        Convert a batch of torch data (i.e. 13D odom tensors)
        into batches of model states
        """
        pass

    @abc.abstractmethod
    def get_actions(self, batch):
        """
        Convert a batch of torch data into batches of actions
        """
        pass
