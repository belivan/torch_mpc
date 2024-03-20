"""
Base class for sampling of actions for sampling MPC. All sampling strategies
must support the following:

Given a nominal sequence of shape B x T x M and action limits (B x M), 
return a Tensor of shape B x N x T x M

where:
    B: number of optimizations (i.e. instances of the MPC problem)
    N: number of samples to draw per optimization
    T: number of timesteps
    M: action dimension

Also, note that the number of samples to produce must be pre-sepcified so
that we can pre-allocate tensors
"""

import abc

class SamplingStrategy(abc.ABC):
    def __init__(self, B, K, H, M, device):
        """
        Args:
            B: Number of optimizations to run
            K: Number of samples to generate per optimization
            H: Number of timesteps
            M: Action dimension
            device: device to compute samples on
        """
        self.B = B
        self.K = K
        self.H = H
        self.M = M
        self.device = device

    @abc.abstractmethod
    def sample(self, u_nominal, u_lb, u_ub):
        """
        Args:
            u_nominal: a [B x T x M] tensor of actions (some strategies will add additive noise to this)
            u_lb: a [B x M] tensor that lower-bounds the action space
            u_ub: a [B x M] tensor that upper-bounds the action space
        """
        pass

    @abc.abstractmethod
    def to(self, device):
        """
        Args:
            device: The device to sample on
        """
        pass
