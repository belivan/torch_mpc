import torch
import numpy as np
import gym

from torch_mpc.models.base import Model
from wheeledsim_rl.util.util import dict_map

class PointMass(Model):
    """
    Pointmass for debugging stuff
    x = [x, y, vx, vy] 
    u = [Fx, Fy]
    xdot = [
        vx
        vy
        Fx/m - kf * |V|
        Fy/m - kf * |V|
    ]
    """
    def __init__(self, m=1.0, kf=1.0, dt=0.1, device='cpu'):
        self.m = m
        self.kf = kf
        self.dt = dt
        self.device = device
        self.u_ub = np.array([1.0, 1.0])
        self.u_lb = np.array([-1.0, -1.0])

    def observation_space(self):
        low = -np.ones(4).astype(float) * float('inf')
        high = -low
        return gym.spaces.Box(low=low, high=high)

    def action_space(self):
        return gym.spaces.Box(low=self.u_lb, high=self.u_ub)

    def dynamics(self, state, action):
        x, y, vx, vy = state.moveaxis(-1, 0)
        fx, fy = action.moveaxis(-1, 0)

        xd = vx
        yd = vy

        #should check that this is actually right
        fric_x = -self.kf * vx 
        fric_y = -self.kf * vy

        vxd = (fx + fric_x) / self.m
        vyd = (fy + fric_y) / self.m

        return torch.stack([xd, yd, vxd, vyd], dim=-1)

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
        return torch.linalg.norm(states[..., 2:], dim=-1)

    def to(self, device):
        self.device = device
        return self

if __name__ == '__main__':
        import matplotlib.pyplot as plt

        pointmass = PointMass()
        x0 = torch.zeros(4)
#        U = torch.ones(100, 2)

        U = torch.stack([
            torch.linspace(0, 10, 100).sin(),
            torch.linspace(0, 10, 100).cos()
        ], axis=-1)

        X = pointmass.rollout(x0, U)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        axs[0].set_title('traj')
        axs[0].plot(X[:, 0], X[:, 1])

        axs[1].set_title('states')
        for i, label in enumerate(['x', 'y', 'vx', 'vy']):
            axs[1].plot(X[:, i], label=label)
        axs[1].legend()

        axs[2].set_title('controls')
        for i, label in enumerate(['fx', 'fy']):
            axs[2].plot(U[:, i], label=label)
        axs[2].legend()

        plt.show()
