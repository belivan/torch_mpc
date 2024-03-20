import torch
import numpy as np

from torch_mpc.models.gravity_throttle_kbm import *
from torch_mpc.util.utils import *

class LinearThrottleKBM(GravityThrottleKBM): 

    def dynamics(self,state, extra_state, action):
        if self.debug:
            import pdb;pdb.set_trace()
        x, y, theta, v, delta = state.moveaxis(-1, 0)
        throttle_og, delta_des_og = action.moveaxis(-1, 0)

        v_actual = v
        delta_actual = delta
        sign_friction = torch.sign(v)
        throttle_net = self.parameters['1'].exp() *throttle_og - self.parameters['2'].exp() * v - self.parameters['3'].exp() * sign_friction
        delta_des = self.parameters['0'].exp() * (delta_des_og - delta)
        center_mask = return_to_center(delta,delta_des_og)
        delta_des[center_mask] *= self.parameters['4'].exp() 
        delta_des = delta_des.clip(-self.steer_rate_lim, self.steer_rate_lim)

        xd = v_actual * theta.cos()
        yd = v_actual * theta.sin()
        thd = v * delta_actual.tan() / self.L
        if thd.max() > 1e5:
            import pdb;pdb.set_trace()
        vd = throttle_net
        deltad = delta_des
        below_steer_lim = (delta < self.steer_lim[0]).float()
        above_steer_lim = (delta > self.steer_lim[1]).float()
        deltad = deltad.clip(0., 1e3) * below_steer_lim + deltad * (1.-below_steer_lim)
        deltad = deltad.clip(-1e3, 0) * above_steer_lim + deltad * (1.-above_steer_lim)
        if self.requires_grad:
            deltad.requires_grad_()
            deltad.register_hook(lambda grad: grad.clamp(max=1e9))		
        return torch.stack([xd, yd, thd, vd, deltad], dim=-1)