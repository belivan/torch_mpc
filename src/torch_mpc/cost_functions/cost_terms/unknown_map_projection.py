import torch

from torch_mpc.cost_functions.cost_terms.base import CostTerm
from torch_mpc.cost_functions.cost_terms.utils import world_to_grid

class UnknownMapProjection:
    """
    Stage cost that projects trajectories onto an unknown map
    """
    def __init__(self, start_idx=0, unknown_threshold=0.05, unknown_penalty=1e8, device='cpu'):
        """
        Args:
            start_idx: don't start counting unknown cells until this point
            unknown_threshold: trajectories that spend more than this amount of time in unknown space are panalized
            unknown_penalty: penalize trajectories in unknown space by this amount
        """
        self.start_idx = start_idx
        self.unknown_threshold = unknown_threshold
        self.unknown_penalty = unknown_penalty
        self.device = device

    def get_data_keys(self):
        return ['unknown_map', 'unknown_map_metadata']

    def cost(self, states, actions, data,past_cost=None,cur_obs = None):
        cost = torch.zeros(states.shape[:2], device=self.device)

        costmap = data['unknown_map']
        metadata = data['unknown_map_metadata']

        world_pos = states[..., :, :2]
        grid_pos, invalid_mask = world_to_grid(world_pos, metadata)

        # Switch grid axes to align with robot centric axes: +x forward, +y left
        grid_pos = grid_pos[..., [1, 0]]

        # Assign invalid costmap indices to a temp value and then set them to invalid cost
        grid_pos[invalid_mask] = 0
        grid_pos = grid_pos.long()
        idx0 = torch.arange(grid_pos.shape[0])
        ndims = len(grid_pos.shape)-2
        idx0 = idx0.view(idx0.shape[0], *[1]*ndims)

        unknowns = torch.clone(costmap[idx0, grid_pos[...,0], grid_pos[...,1]])
        unknowns[invalid_mask] = 1

        is_unknown_mask = unknowns[:, :, self.start_idx:].mean(dim=-1) > self.unknown_threshold

        cost[is_unknown_mask] = self.unknown_penalty

        return cost

    def to(self, device):
        self.device = device
        return self

    def __repr__(self):
        return "Unknown Map Projection"
