"""
Base class for update rules. These will take in:
    1. A [B x N x T x M] tensor of action sequences
    2. A [B x N] tensor of costs associated with each sequence

and return a [B x T x M] action sequence to execute

(note that we'll probably need to return additional stuff i.e. CMA-ES variance updates)
"""

import abc

class UpdateRule(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def update(self, action_sequences, costs):
        """
        Args:
            action_sequences: a [B x N x T x M] Tensor of controls
            costs: a [B x N] Tensor of costs
        """
        pass
