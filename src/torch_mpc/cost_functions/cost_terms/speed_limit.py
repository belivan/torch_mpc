import torch

from torch_mpc.cost_functions.cost_terms.base import CostTerm

class SpeedLimit:
    """
    Cost term that enforces speed limits on the model
    """
    def __init__(self, target_speed=4., max_speed=5., sharpness=10., speed_idx=3, device='cpu'):
        """
        Args:
            target_speed: desired speed (can go a bit faster than this)
            max_speed: upper limit on speed (will not go faster than this)
            sharpness: multiply amt over target by this before exponentiating
            speed_idx: the index of the speed variable in the model
        """
        self.target_speed = target_speed
        self.max_speed = max_speed
        self.sharpness = sharpness
        self.speed_idx = speed_idx
        self.device = device

    def get_data_keys(self):
        return []

    def cost(self, states, actions, feasible, data):
        speed = states[..., self.speed_idx].abs()
        cost = torch.zeros(speed.shape[0], speed.shape[1], device=speed.device)

        within_speed_limit = speed <= self.max_speed
        new_feasible = within_speed_limit.all(dim=-1)

        cost = (self.sharpness * (speed - self.target_speed)).exp().mean(dim=-1)
        
        return cost, new_feasible

    def to(self, device):
        self.device = device
        return self

    def __repr__(self):
        return "Speed Limit"
