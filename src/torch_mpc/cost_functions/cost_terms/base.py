"""
Base class for cost function terms. At a high-level, cost function terms need to be able to do the following:
    1. Tell the high-level cost manager what data it needs to compute costs
    2. Actually compute cost values given:
        a. A [B1 x B2 x T x N] tensor of states
        b. A [B1 x B2 x T x M] tensor of actions

    Note that we assume two batch dimensions as we often want to perform multiple sampling-based opts in parallel
"""

import abc

class CostTerm(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_data_keys(self):
        """
        Returns:
            the set of keys that the cost combiner is expected to give it
        """
        pass

    @abc.abstractmethod
    def cost(self, states, actions, feasible, data):
        """
        Args:
            states: a [B1 x B2 x T x N] tensor of states
            actions: a [B x B2 x T x M] tensor of actions
            feasible: a [B x B2] tensor of feasibility
            data: a *pointer* to the data from the high-level cost function
        """
        pass
