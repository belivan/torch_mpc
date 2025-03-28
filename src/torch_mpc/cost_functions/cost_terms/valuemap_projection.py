import numpy as np
import torch

from torch_mpc.cost_functions.cost_terms.base import CostTerm
from torch_mpc.cost_functions.cost_terms.utils import world_to_grid, move_to_local_frame

class ValuemapProjection:
    """
    Stage/terminal cost that leverages a value function
    """
    def __init__(self, length=5.0, width=3.0, nl=3, nw=3, length_offset=-1.0, width_offset=0., local_frame=False, valuemap_key='local_valuemap', final_state=False, device='cpu'):
        """
        Args:
            length: the length of the footprint (along x)
            width: the width of the footprint (along y)
            nl: number of sample points along y
            nw: number of sample points along x
            length_offset: move the footprint length-wise by this amount
            width_offset: move the footprint width-wise by this amount
            valuemap_key: the key to look for valuemap data in
            final_state: if true, only evaluate at final state, else all
        """
        self.length = length
        self.width = width
        self.length_offset = length_offset
        self.width_offset = width_offset
        self.nl = nl
        self.nw = nw
        self.local_frame = local_frame
        self.valuemap_key = valuemap_key
        self.final_state = final_state
        self.device = device
        self.footprint = self.make_footprint()

    def make_footprint(self):
        xs = torch.linspace(-self.length/2., self.length/2., self.nl, device=self.device) + self.length_offset
        ys = torch.linspace(-self.width/2., self.width/2., self.nw, device=self.device) + self.width_offset
        footprint = torch.stack(torch.meshgrid(xs, ys, indexing='ij'), dim=-1).view(-1, 2)
        return footprint

    def apply_footprint(self, traj):
        """
        Given a B x K x T x 3 tensor of states (last dim is [x, y, th]),
        return a B x K x T x F x 2 tensor of positions (F is each footprint sample)
        """
        tdims = traj.shape[:-1]
        nf = self.footprint.shape[0]

        pos = traj[..., :2]
        th = traj[..., 2]

        R = torch.stack([
            torch.stack([th.cos(), -th.sin()], dim=-1),
            torch.stack([th.sin(), th.cos()], dim=-1),
        ], dim=-2) #[B x K x T x 2 x 2]

        R_expand = R.view(*tdims, 1, 2, 2) #[B x K x T x F x 2 x 2]
        footprint_expand = self.footprint.view(1, 1, 1, nf, 2, 1) #[B x K x T x F x 2 x 1]

        footprint_rot = (R_expand @ footprint_expand).view(*tdims, nf, 2) #[B x K x T X F x 2]
        footprint_traj = pos.view(*tdims, 1, 2) + footprint_rot

#        #debug viz
#        import matplotlib.pyplot as plt
#        for i in range(footprint_traj.shape[1]):
#            tr = traj[0, i] #[T x 3]
#            ftr = footprint_traj[0, i].view(-1, 2)
#            plt.plot(tr[:, 0], tr[:, 1], c='r')
#            plt.scatter(ftr[:, 0], ftr[:, 1], c='b', alpha=0.1)
#            plt.gca().set_aspect(1.)
#            plt.show()

        return footprint_traj

    def get_data_keys(self):
        return [self.valuemap_key]

    def cost(self, states, actions, feasible, data):
        if self.local_frame:
            states2 = move_to_local_frame(states)
        else:
            states2 = states

        cost = torch.zeros(states2.shape[:2], device=self.device)

        costmap = data[self.valuemap_key]['data']
        metadata = data[self.valuemap_key]['metadata']

        world_pos = states2[..., :, :3] # x, y, th
        footprint_pos = self.apply_footprint(world_pos) #[B x K x T x F x 2]
        grid_pos, invalid_mask = world_to_grid(footprint_pos, metadata)

        yaws = states2[..., :, 2] % (2*np.pi)
        ang_targets = metadata['headings'].squeeze()
        ang_targets = torch.cat([ang_targets, torch.tensor([2*np.pi], device=ang_targets.device)])
        _ldims = [1] * len(yaws.shape)
        angdiff = (yaws.unsqueeze(-1) - ang_targets.view(*_ldims, -1)).abs()
        angidx = angdiff.argmin(dim=-1) % costmap.shape[-1]
        angidx = torch.stack([angidx]*len(self.footprint), dim=-1)

        # Switch grid axes to align with robot centric axes: +x forward, +y left
#        grid_pos = grid_pos[..., [1, 0]]

        # Assign invalid costmap indices to a temp value and then set them to invalid cost
        grid_pos[invalid_mask] = 0
        grid_pos = grid_pos.long()
        idx0 = torch.arange(grid_pos.shape[0])
        ndims = len(grid_pos.shape)-2
        idx0 = idx0.view(idx0.shape[0], *[1]*ndims)

        new_costs = torch.clone(costmap[idx0, grid_pos[...,0], grid_pos[...,1], angidx])
        new_costs[invalid_mask] = 0.

        new_feasible = feasible.clone()

        #avg cost over footprint, sum over time
        if self.final_state:
            cost += new_costs.mean(dim=-1)[..., -1]
        else:
            # in this case, mean is the correct reduction to preserve cost/cost-to-go weighting
            cost += new_costs.mean(dim=-1).mean(dim=-1)

        return cost, new_feasible

    def to(self, device):
        self.footprint = self.footprint.to(device)
        self.device = device
        return self

    def __repr__(self):
        return "Costmap Projection"
