import torch
import numpy as np
import gym

from torch_mpc.models.base import Model
from wheeledsim_rl.util.util import dict_map

class KBM(Model):
    """
    Kinematic bicycle model
    x = [x, y, th] 
    u = [v, delta]
    xdot = [
        v * cos(th)
        v * sin(th)
        L / tan(delta)
    ]
    """
    def __init__(self, L=3.0, min_throttle=0., max_throttle=1., max_steer=0.3, dt=0.1, device='cpu'):
        self.L = L
        self.dt = dt
        self.device = device
        self.u_ub = np.array([max_throttle, max_steer])
        self.u_lb = np.array([min_throttle, -max_steer])

    def observation_space(self):
        low = -np.ones(3).astype(float) * float('inf')
        high = -low
        return gym.spaces.Box(low=low, high=high)

    def action_space(self):
        return gym.spaces.Box(low=self.u_lb, high=self.u_ub)

    def dynamics(self, state, action):
        x, y, th = state.moveaxis(-1, 0)
        v, d = action.moveaxis(-1, 0)
        xd = v * th.cos()
        yd = v * th.sin()
        thd = v * d.tan() / self.L
        return torch.stack([xd, yd, thd], dim=-1)

    def predict(self, state, action):
        k1 = self.dynamics(state, action)
        k2 = self.dynamics(state + (self.dt/2)*k1, action)
        k3 = self.dynamics(state + (self.dt/2)*k2, action)
        k4 = self.dynamics(state + self.dt*k3, action)

        next_state = state + (self.dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        return next_state

    def rollout(self, state, actions):
        """
        Expected shapes:
            state: [B1 x ... x Bn x xd]
            actions: [B1 x ... x Bn x T x ud]
            returns: [B1 x ... x Bn X T x xd]
        """
        X = []
        curr_state = state
        for t in range(actions.shape[-2]):
            action = actions[..., t, :]
            next_state = self.predict(curr_state, action)
            X.append(next_state)
            curr_state = next_state.clone()
        
        return torch.stack(X, dim=-2)

    def quat_to_yaw(self, q):
        #quats are x,y,z,w
        qx, qy, qz, qw = q.moveaxis(-1, 0)
        return torch.atan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))

    def get_observations(self, batch):
        state = batch['state']
        if len(state.shape) == 1:
            return self.get_observations(dict_map(batch, lambda x:x.unsqueeze(0))).squeeze()

        x = state[..., 0]
        y = state[..., 1]
        q = state[..., 3:7]
        yaw = self.quat_to_yaw(q)
        return torch.stack([x, y, yaw], axis=-1)

    def get_actions(self, batch):
        return batch

    def get_speed(self, states):
        raise NotImplementedError

    def to(self, device):
        self.device = device
        return self

class SteerTargetKBM(KBM):  # not being used
    """
    KBM, but the states, actions are now
    x = [x, y, th, delta] 
    u = [v, target]
    xdot = [
        v * cos(th)
        v * sin(th)
        L / tan(delta)
        Kv (delta - target)
    ]
    """
    def __init__(self, L=3.0, min_throttle=0., max_throttle=1., max_steer=0.3, steer_rate=1.0, Ks=1.0, dt=0.1, steer_to_delta=(30./415.), device='cpu'):
        self.L = L
        self.dt = dt
        self.device = device
        self.u_lb = np.array([min_throttle, -max_steer])
        self.u_ub = np.array([max_throttle, max_steer])
        self.steer_rate = steer_rate
        self.Ks = Ks
        self.steer_to_delta = steer_to_delta

    def observation_space(self):
        low = -np.ones(4).astype(float) * float('inf')
        high = -low
        return gym.spaces.Box(low=low, high=high)

    def action_space(self):
        return gym.spaces.Box(low=self.u_lb, high=self.u_ub)

    def dynamics(self, state, action):
        x, y, th, d = state.moveaxis(-1, 0)
        v, target = action.moveaxis(-1, 0)
        xd = v * th.cos()
        yd = v * th.sin()
        thd = v * d.tan() / self.L
        dd = self.Ks * (target - d)
        dd[dd > self.steer_rate] = self.steer_rate
        dd[dd < -self.steer_rate] = -self.steer_rate
        return torch.stack([xd, yd, thd, dd], dim=-1)

    def get_observations(self, batch):
        state = batch['state']
        if len(state.shape) == 1:
            return self.get_observations(dict_map(batch, lambda x:x.unsqueeze(0))).squeeze()

        x = state[..., 0]
        y = state[..., 1]
        q = state[..., 3:7]
        yaw = self.quat_to_yaw(q)

        steer_angle = batch['steer_angle']
        steer_angle_rad = steer_angle * (np.pi/180.) * self.steer_to_delta
        return torch.stack([x, y, yaw, steer_angle_rad], axis=-1)
